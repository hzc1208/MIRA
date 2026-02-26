import torch
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from vllm import LLM, SamplingParams
from clip_model import load_video_or_frames
from typing import List
import os, sys, time, json, pickle
import traceback


class CaptionModel:
    def __init__(self, model_name: str, gpu_num: int, max_new_tokens: int = 256):
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_mm_preprocessor_cache=True,
            max_model_len=8192,
            gpu_memory_utilization=0.75,
            max_num_seqs=32,
            tensor_parallel_size=gpu_num,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 128 * 28 * 28
            }
        )
        self.rules = "The description of image must abide by the following policies:\n" + \
            "    1. You may describe the foreground / background / salient objects. \n" + \
            "    2. When describing objects, please endeavor to include as much of the following information:\n" + \
            "        2.1. textures / attributes / locations / presence / status / characteristics / numbers of objects\n" + \
            "        2.2. relative positions between objects\n" + \
            "    3. If there are commen sense or world knowledge, for example, species, celebrities, scenic spots and historical sites, you must state them explicitly instead of using phrases like \"a person\", \"a place\", etc.\n" + \
            "    4. Other objective and subjective details that can help understand and reproduce the image.\n" + \
            "    5. Text contents must be appeared in the description if there exists. Keep the original language of text content.\n" + \
            "    6. The description should NO longer than 25 words, keep the description in a single paragraph.\n" +\
            "    7. The description should NOT start with a form such as 'this image depicts...', please use a narrative statement.\n"
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            top_k=20,
            max_tokens=max_new_tokens
        )

    def generate_prompt(self, query: str):
        return (
            f"<|im_start|>system\n{self.rules}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"Given the user query: {query}, please describe the content of this image. "
            f"If there is any content related to the user query, it should be described and emphasized in detail. <|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate_caption(self, frames: List[Image.Image], query: str, batch_size: int = 32) -> List[str]:
        inputs, output_texts = [], []
        for img in frames:
            inputs.append({
                "prompt": self.generate_prompt(query),
                "multi_modal_data": {
                    "image": img
                },
            })
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            outputs = self.model.generate(batch_inputs, sampling_params=self.sampling_params)
            output_texts.extend([output.outputs[0].text.strip() for output in outputs])

        return output_texts


def generate_captions_for_subscenes(
    frames_pil: List[Image.Image],
    video_query: str,
    video_info: dict,
    model: CaptionModel,
    K: int,
    window_size: int = 2
):
    if ("frames_for_cap" in video_info) and (len(video_info["frames_for_cap"]) > 0):
        frames_for_cap = video_info["frames_for_cap"]
        final_caps = model.generate_caption([frames_pil[i] for i in frames_for_cap], video_query)
        video_info["index_to_caption"] = {
            frames_for_cap[i]: final_caps[i] for i in range(len(frames_for_cap))
        }
        del video_info["frames_for_cap"]
        return video_info

    index_to_score, index_to_subscene, subscene_to_index = video_info["index_to_score"], video_info["index_to_subscene"], video_info["subscene_to_index"]
    subscene_scores = {
        sub_idx: 0.0 for sub_idx in subscene_to_index
    }
    subscene_max_score_indices = {}
    top_k_relevant_indices = sorted(index_to_score, key=lambda k: index_to_score[k], reverse=True)
    
    for i in range(len(top_k_relevant_indices)):
        subscene_index = index_to_subscene[top_k_relevant_indices[i]]
        subscene_scores[subscene_index] += index_to_score[top_k_relevant_indices[i]]
        if subscene_index not in subscene_max_score_indices:
            subscene_max_score_indices[subscene_index] = top_k_relevant_indices[i]

    top_k_relevant_subscenes = [sub_idx for sub_idx in sorted(subscene_scores, key=lambda k: subscene_scores[k], reverse=True)[:K]]
    subscenes_for_cap, frames_for_cap_dict = [], {}
    for subscene_index in top_k_relevant_subscenes:
        top_k_neighbor_subscenes = np.arange(max(0, subscene_index - window_size), min(len(subscene_to_index), subscene_index + window_size + 1)).tolist()
        if len(top_k_neighbor_subscenes) > 3:
            subscenes_for_cap.extend(top_k_neighbor_subscenes)
            frames_for_cap_dict[subscene_max_score_indices[subscene_index]] = [subscene_max_score_indices[i] for i in top_k_neighbor_subscenes if i != subscene_index]

    if len(subscenes_for_cap) > 0:
        subscenes_for_cap = set(subscenes_for_cap)
        frames_for_cap = sorted([subscene_max_score_indices[i] for i in subscenes_for_cap])
        final_caps = model.generate_caption([frames_pil[i] for i in frames_for_cap], video_query)
        video_info["index_to_caption"] = {
            frames_for_cap[i]: final_caps[i] for i in range(len(frames_for_cap))
        }
        video_info["frames_for_cap_dict"] = frames_for_cap_dict
    else:
        video_info["index_to_caption"], video_info["frames_for_cap_dict"] = {}, {}

    return video_info


def process_video(task, shared_model, args):
    try:
        video_index, video_path, video_query, video_info = task['video_index'], task['video_path'], task['video_question'], task['video_info']
        if ("frames_for_cap" in video_info) and (len(video_info["frames_for_cap"]) > 0):
            fn_for_cap_length = len(video_info["frames_for_cap"])
            args.logger.info(f"[INFO] Processing video {video_index}: {video_path} (w/ gw), {fn_for_cap_length} frames for cap")
        # Step 1: Load frames
        frames_pil, frame_indices = load_video_or_frames(args, video_path, args.keyframe_num + args.causalframe_num)
        # args.logger.info(f"[INFO] Loaded {len(frames_pil)} frames from video {video_index}.")
        global_start_time = time.time()
        # Step 2: Generate subscene captions
        video_info = generate_captions_for_subscenes(frames_pil, video_query, video_info, shared_model, args.causalframe_num)

        return (video_index, video_info), {"total_time": time.time() - global_start_time}
    except Exception as e:
        args.logger.error(f"[ERROR] Failed to process video {video_index}: {str(e)}")
        args.logger.error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        return None, None


def init_cap_model(model_root, gpu_num):
    torch.cuda.empty_cache()
    return CaptionModel(model_root, gpu_num=gpu_num)


def run_cap_on_gpu(my_tasks, args, gpu_indices):
    if len(args.gpus) > 1:
        device_id = ','.join(map(str, gpu_indices))
    else:
        device_id = str(args.gpus[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    save_dicts, time_logs = [], []
    # Filter tasks assigned to this GPU
    if not my_tasks:
        args.logger.info(f"[INFO] No tasks assigned to GPU {device_id}, skipping.")
        return
    args.logger.info(f"[INFO] Starting {len(my_tasks)} tasks on GPU {device_id}")

    # Load model on GPUs
    shared_model = init_cap_model(args.vllm_root, len(gpu_indices))
    for task in my_tasks:
        save_dict = None
        while save_dict is None:
            save_dict, save_times = process_video(task, shared_model, args)
        save_dicts.append(save_dict)
        time_logs.append(save_times)

    return save_dicts, time_logs


def run_cap_on_gpus(all_tasks, args):
    num_process = max(len(args.gpus), 1)
    save_dicts, all_time_logs = [], []
    # Group tasks by GPU assignment
    gpu_task_groups = [[] for _ in range(num_process)]
    for i, task in enumerate(all_tasks):
        process_idx = i % num_process
        gpu_task_groups[process_idx].append(task)

    with ProcessPoolExecutor(max_workers=num_process) as executor:
        futures = [executor.submit(run_cap_on_gpu, gpu_task_groups[i], args, args.gpus[i:i+1]) for i in range(num_process)]
        for future in futures:
            result, time_logs = future.result()
            save_dicts.extend(result)
            all_time_logs.extend(time_logs)

    return sorted(save_dicts, key=lambda x: x[0]), all_time_logs


def format_question_with_choices(question: str, choices: List[str]) -> str:
    result = f"{question}\n Optional choices:"
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        result += f"\n {letter}: {choice}"    
    return result


def run_cap_main(args):
    if args.dataset_name == "longvideobench":
       label_dir = os.path.join(args.dataset_dir,'lvb_val.json')
       video_dir = os.path.join(args.dataset_dir,'videos')
    elif args.dataset_name == "videomme":
       label_dir = os.path.join(args.dataset_dir,'videomme.json')
       video_dir = os.path.join(args.dataset_dir,'data')
    elif args.dataset_name == "mlvu":
       label_dir = os.path.join(args.dataset_dir,'base.json')
       video_dir = os.path.join(args.dataset_dir,'videos')
    elif args.dataset_name == "egoschema":
       label_dir = os.path.join(args.dataset_dir,'base.json')
       video_dir = os.path.join(args.dataset_dir,'videos')
    else:
       raise ValueError("dataset_name: longvideobench, videomme, mlvu, egoschema")

    if os.path.exists(label_dir):
        with open(label_dir,'r') as f:
            video_datas = json.load(f)
        if args.dataset_name == "mlvu":
            video_datas = [video_data for video_data in video_datas if video_data.get("candidates") is not None]
    else:
        raise OSError("the label file does not exist")

    if os.path.exists(args.info_dir):
        with open(args.info_dir,'rb') as f:
            video_data_infos = pickle.load(f)
    else:
        raise OSError("the info file does not exist") 

    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset_name == "longvideobench":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["video_path"]),
                'video_question': format_question_with_choices(video_data["question"], video_data["candidates"]),
                'video_info': video_data_infos[i]
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "videomme":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["videoID"]+'.mp4'),
                'video_question': format_question_with_choices(video_data["question"], video_data["options"]),
                'video_info': video_data_infos[i]
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "mlvu":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["question_type"], video_data["video"]),
                'video_question': format_question_with_choices(video_data["question"], video_data["candidates"]),
                'video_info': video_data_infos[i]
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "egoschema":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["video_idx"]+'.mp4'),
                'video_question': format_question_with_choices(video_data["question"], video_data["option"]),
                'video_info': video_data_infos[i]
            } for i, video_data in enumerate(video_datas)
        ]

    save_dicts, all_time_logs = run_cap_on_gpus(all_tasks, args)
    total_time = sum(t["total_time"] for t in all_time_logs) / len(all_time_logs)
    args.logger.info(f"[INFO] Total time taken (cap): {total_time:.2f} seconds")
    
    save_lists = []
    for video_index, save_dict in save_dicts:
        save_lists.append(save_dict)

    with open(os.path.join(args.output_dir, f"{args.dataset_name}_clip_info.pkl"), 'wb') as f:
        pickle.dump(save_lists, f)
    args.info_dir = os.path.join(args.output_dir, f"{args.dataset_name}_clip_info.pkl")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from clip_model import extract_batch_features
from typing import List, Tuple
import os, sys, time, json, pickle, math


class CausalModel:
    def __init__(self, model_name: str, gpu_num: int, gpu_memory_utilization: float = 0.5, max_new_tokens: int = 256):
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=gpu_num
        )
        self.rules = "You are the most powerful language model for causal reasoning. For a video, given the visual descriptions corresponding to some of the scenes at certain moments, " + \
            "please infer the most likely visual description corresponding to the scene at the time I specified based on the continuity and causality of the event development.\n\n" + \
            "The description must abide by the following policies:\n" + \
            "    1. The form and length of the description should be consistent with the description form and length corresponding to other given moments. " + \
            "The specific time (e.g. At Time xxx) should NOT appear in the description. The description should usually NO longer than 25 words.\n"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            top_k=20,
            max_tokens=max_new_tokens
        )

    def generate_caption(self, contexts: List[List[str]], inf_moments: List[List[int]], batch_size: int = 4) -> List[str]:
        output_texts = []
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i + batch_size]
            batch_inf_moments = inf_moments[i:i + batch_size]
            inputs = [[
                {"role": "system", "content": self.rules},
                {"role": "user", "content": f"Given the video scene description corresponding to the following moments:\n{context}\n"
                f"What is the most likely visual description corresponding to the scene at time {inf_moment}?"}]
                for batch_context, batch_inf_moment in zip(batch_contexts, batch_inf_moments)
                for context, inf_moment in zip(batch_context, batch_inf_moment)]
            inputs = self.tokenizer.apply_chat_template(inputs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            inputs = [TokensPrompt(prompt_token_ids=ele) for ele in inputs]
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
            output_texts.extend([output.outputs[0].text.strip() for output in outputs])
        return output_texts


def infer_causal_frames(
    video_info: dict,
    video_query: str,
    model: CausalModel,
    txt_evaluator: tuple
):
    index_to_caption, frames_for_cap_dict, index_to_frame_idx = video_info["index_to_caption"], video_info["frames_for_cap_dict"], video_info["index_to_frame_idx"]
    if len(index_to_caption) > 0:
        contexts, inf_moments, ref_moments = [], [], []
        for inf_index, neigh_indices in frames_for_cap_dict.items():
            ctx_indices_dict = {
                ref_idx : [idx for idx in neigh_indices if idx != ref_idx] for ref_idx in neigh_indices
            }
            context, inf_moment = [], []
            for ref_idx, ctx_indices in ctx_indices_dict.items():
                context.append(
                    "\n".join([f"Time {idx}: {index_to_caption.get(idx, 'No description available')}" for idx in ctx_indices])
                )
                inf_moment.append(inf_index)
                ref_moments.append(ref_idx)
            contexts.append(context)
            inf_moments.append(inf_moment)

        inf_txt_features = extract_batch_features(
            txt_evaluator[0], None, txt_evaluator[1], texts=model.generate_caption(contexts, inf_moments), choice="text"
        )    
        inf_txt_gt_features = extract_batch_features(
            txt_evaluator[0], None, txt_evaluator[1], texts=[index_to_caption[i] for i in \
            [inf_idx for inf_moment in inf_moments for inf_idx in inf_moment]], choice="text"
        )
        index_to_causal_score, index_to_cap_relevant_score = {index: 0.0 for index in set(ref_moments)}, dict()
        for i in range(len(ref_moments)):
            cos_sim = torch.matmul(inf_txt_features[i], inf_txt_gt_features[i].t()).item()
            index_to_causal_score[ref_moments[i]] += math.sqrt(max(0.0, 1 - cos_sim**2))

        del inf_txt_features, inf_txt_gt_features
        video_query_feature = extract_batch_features(
            txt_evaluator[0], None, txt_evaluator[1], texts=[video_query], choice="text"
        )
        video_cap_features = extract_batch_features(
            txt_evaluator[0], None, txt_evaluator[1], texts=[index_to_caption[i] for i in \
            sorted(index_to_caption.keys())], choice="text"
        )
        for i, cap_idx in enumerate(sorted(index_to_caption.keys())):
            if cap_idx not in index_to_cap_relevant_score:
                index_to_cap_relevant_score[cap_idx] = torch.matmul(video_cap_features[i], video_query_feature.t()).item()

        return {
            "index_to_frame_idx": index_to_frame_idx,
            "index_to_causal_score": index_to_causal_score,
            "index_to_cap_relevant_score": index_to_cap_relevant_score
        }
    else:
        return {
            "index_to_frame_idx": {},
            "index_to_causal_score": {},
            "index_to_cap_relevant_score": {}
        }


def process_video(task, shared_model, args):
    global_start_time = time.time()
    video_index, video_info, video_query = task['video_index'], task['video_info'], task['video_question']
    model, txt_evaluator = shared_model

    # Step 1: Return top-K frames based on causal inference
    causal_frame_indices = infer_causal_frames(video_info, video_query, model, txt_evaluator)
    args.logger.info(f"[INFO] Processing video {video_index}: {len(causal_frame_indices['index_to_causal_score'])} causal frames")

    return (video_index, causal_frame_indices), {"total_time": time.time() - global_start_time}


def init_causal_model(model_root: Tuple[str, str], gpu_num):
    torch.cuda.empty_cache()
    llm = CausalModel(model_root[0], gpu_num=gpu_num)
    txt_model = AutoModelForCausalLM.from_pretrained(model_root[1], trust_remote_code=True, device_map="auto").eval()
    txt_tokenizer = AutoTokenizer.from_pretrained(model_root[1])
    return (llm, (txt_model, txt_tokenizer))


def run_causal_on_gpu(my_tasks, args, gpu_indices):
    if len(args.gpus) > 1:
        device_id = ','.join(map(str, gpu_indices))
    else:
        device_id = str(args.gpus[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    results, time_logs = [], []
    # Filter tasks assigned to this GPU
    if not my_tasks:
        args.logger.info(f"[INFO] No tasks assigned to GPU {device_id}, skipping.")
        return
    args.logger.info(f"[INFO] Starting {len(my_tasks)} tasks on GPU {device_id}")

    # Load model on GPUs
    shared_model = init_causal_model((args.llm_root, args.clip_root), len(gpu_indices))
    for task in my_tasks:
        result, save_times = process_video(task, shared_model, args)
        results.append(result)
        time_logs.append(save_times)

    return results, time_logs


def run_causal_on_gpus(all_tasks, args):
    num_process = max(len(args.gpus) // 2, 1)
    results, all_time_logs = [], []
    # Group tasks by GPU assignment
    gpu_task_groups = [[] for _ in range(num_process)]
    for i, task in enumerate(all_tasks):
        process_idx = i % num_process
        gpu_task_groups[process_idx].append(task)

    with ProcessPoolExecutor(max_workers=num_process) as executor:
        futures = [executor.submit(run_causal_on_gpu, gpu_task_groups[i], args, args.gpus[i*2:(i+1)*2]) for i in range(num_process)]
        for future in futures:
            result, time_logs = future.result()
            results.extend(result)
            all_time_logs.extend(time_logs)

    return sorted(results, key=lambda x: x[0]), all_time_logs


def run_causal_main(args):
    if args.dataset_name =="longvideobench":
       label_dir = os.path.join(args.dataset_dir,'lvb_val.json')
    elif args.dataset_name =="videomme":
       label_dir = os.path.join(args.dataset_dir,'videomme.json')
    elif args.dataset_name == "mlvu":
       label_dir = os.path.join(args.dataset_dir,'base.json')
    elif args.dataset_name == "egoschema":
       label_dir = os.path.join(args.dataset_dir,'base.json')
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

    if args.dataset_name in ["longvideobench", "mlvu"]:
        all_tasks = [
            {
                'video_index': i,
                'video_info': video_data_info,
                'video_question': video_datas[i]["question"] + ','.join(video_datas[i]["candidates"])
            } for i, video_data_info in enumerate(video_data_infos)
        ]
    elif args.dataset_name == "videomme":
        all_tasks = [
            {
                'video_index': i,
                'video_info': video_data_info,
                'video_question': video_datas[i]["question"] + ','.join(video_datas[i]["options"])
            } for i, video_data_info in enumerate(video_data_infos)
        ]
    elif args.dataset_name == "egoschema":
        all_tasks = [
            {
                'video_index': i,
                'video_info': video_data_info,
                'video_question': video_datas[i]["question"] + ','.join(video_datas[i]["option"])
            } for i, video_data_info in enumerate(video_data_infos)
        ]

    dataset_causal_dicts, all_time_logs = run_causal_on_gpus(all_tasks, args)
    total_time = sum(t["total_time"] for t in all_time_logs) / len(all_time_logs)
    args.logger.info(f"[INFO] Total time taken (causal): {total_time:.2f} seconds")
    
    dataset_causal_lists = []
    for video_index, res_frame_indices in dataset_causal_dicts:
        dataset_causal_lists.append(res_frame_indices)

    with open(os.path.join(args.output_dir, f"{args.dataset_name}_causal_frames.pkl"), 'wb') as f:
        pickle.dump(dataset_causal_lists, f)

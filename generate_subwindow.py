import torch
from transformers import AutoImageProcessor, AutoModelForCausalLM
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from vllm import LLM, SamplingParams
from clip_model import load_video_or_frames, extract_batch_features, compute_pairwise_similarity
from typing import List
import os, sys, time, json, pickle
import traceback


def generate_windows_for_subscenes(
    frames_pil: List[Image.Image],
    video_info: dict,
    shared_model,
    K: int,
    window_size: int = 2
):
    model, image_processor = shared_model
    index_to_score, index_to_subscene, subscene_to_index = video_info["index_to_score"], video_info["index_to_subscene"], video_info["subscene_to_index"]
    subscene_scores = {
        sub_idx: 0.0 for sub_idx in subscene_to_index
    }
    subscene_max_score_indices = {}
    top_k_relevant_indices = sorted(index_to_score, key=lambda k: index_to_score[k], reverse=True)
    indices_for_clip_sim, frames_for_cap, frames_for_cap_dict = [], [], {}
    
    for i in range(len(top_k_relevant_indices)):
        subscene_index = index_to_subscene[top_k_relevant_indices[i]]
        subscene_scores[subscene_index] += index_to_score[top_k_relevant_indices[i]]
        if subscene_index not in subscene_max_score_indices:
            subscene_max_score_indices[subscene_index] = top_k_relevant_indices[i]
            indices_for_clip_sim.append(top_k_relevant_indices[i])
    
    indices_for_clip_sim = sorted(indices_for_clip_sim)
    img_features = extract_batch_features(
        model, image_processor, None, images=[frames_pil[i] for i in indices_for_clip_sim], choice="image"
    )
    i2i_sim_matrix = compute_pairwise_similarity(img_features, None)
    indices_to_matrix_row_id = {index : i for i, index in enumerate(indices_for_clip_sim)}
    top_k_relevant_subscenes = [sub_idx for sub_idx in sorted(subscene_scores, key=lambda k: subscene_scores[k], reverse=True)[:K]]
    for subscene_index in top_k_relevant_subscenes:
        anchor_index = subscene_max_score_indices[subscene_index]
        top_k_matrix_row_id = torch.argsort(i2i_sim_matrix[indices_to_matrix_row_id[anchor_index]], descending=True)[1:]
        neigh_counter, neigh_indices = {'ahead': 0, 'behind': 0}, []
        for i in top_k_matrix_row_id:
            neigh_index = indices_for_clip_sim[i]
            if index_to_score[neigh_index] > index_to_score[anchor_index]:
                continue
            if (neigh_index > anchor_index) and (neigh_counter['behind'] < window_size):
                neigh_indices.append(neigh_index)
                neigh_counter['behind'] += 1
            elif (neigh_index < anchor_index) and (neigh_counter['ahead'] < window_size):
                neigh_indices.append(neigh_index)
                neigh_counter['ahead'] += 1
            if len(neigh_indices) == (2 * window_size):
                break

        if len(neigh_indices) > 3:
            frames_for_cap.append(anchor_index)
            frames_for_cap.extend(neigh_indices)
            frames_for_cap_dict[anchor_index] = sorted(neigh_indices)

    video_info["frames_for_cap"] = sorted(list(set(frames_for_cap)))
    video_info["frames_for_cap_dict"] = frames_for_cap_dict

    return video_info


def process_video(task, shared_model, args):
    try:
        global_start_time = time.time()
        video_index, video_path, video_info = task['video_index'], task['video_path'], task['video_info']
        # Step 1: Load frames
        frames_pil, frame_indices = load_video_or_frames(args, video_path, args.keyframe_num + args.causalframe_num)
        args.logger.info(f"[INFO] Loaded {len(frames_pil)} frames from video {video_index}.")
        # Step 2: Generate subscene windows
        video_info = generate_windows_for_subscenes(frames_pil, video_info, shared_model, args.causalframe_num)

        return (video_index, video_info), {"total_time": time.time() - global_start_time}
    except Exception as e:
        args.logger.error(f"[ERROR] Failed to process video {video_index}: {str(e)}")
        args.logger.error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        return None, None


def init_clip_model(model_root):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).to("cuda").eval()
    image_processor = AutoImageProcessor.from_pretrained(model_root)
    return (model, image_processor)


def run_gw_on_gpu(my_tasks, args, gpu_idx):
    device_id = args.gpus[gpu_idx]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    save_dicts, time_logs = [], []
    # Filter tasks assigned to this GPU
    if not my_tasks:
        args.logger.info(f"[INFO] No tasks assigned to GPU {device_id}, skipping.")
        return
    args.logger.info(f"[INFO] Starting {len(my_tasks)} tasks on GPU {device_id}")

    # Load model on GPUs
    shared_model = init_clip_model(args.clip_root)
    for task in my_tasks:
        save_dict = None
        while save_dict is None:
            save_dict, save_times = process_video(task, shared_model, args)
        save_dicts.append(save_dict)
        time_logs.append(save_times)

    return save_dicts, time_logs


def run_gw_on_gpus(all_tasks, args):
    num_process = max(len(args.gpus), 1)
    save_dicts, all_time_logs = [], []
    # Group tasks by GPU assignment
    gpu_task_groups = [[] for _ in range(num_process)]
    for i, task in enumerate(all_tasks):
        process_idx = i % num_process
        gpu_task_groups[process_idx].append(task)

    with ProcessPoolExecutor(max_workers=num_process) as executor:
        futures = [executor.submit(run_gw_on_gpu, gpu_task_groups[i], args, i) for i in range(num_process)]
        for future in futures:
            result, time_logs = future.result()
            save_dicts.extend(result)
            all_time_logs.extend(time_logs)

    return sorted(save_dicts, key=lambda x: x[0]), all_time_logs


def run_gw_main(args):
    if args.dataset_name == "longvideobench":
       label_dir = os.path.join(args.dataset_dir,'lvb_val.json')
       video_dir = os.path.join(args.dataset_dir,'videos')
    elif args.dataset_name == "videomme":
       label_dir = os.path.join(args.dataset_dir,'videomme.json')
       video_dir = os.path.join(args.dataset_dir,'data')
    elif args.dataset_name == "mlvu":
       label_dir = os.path.join(args.dataset_dir,'base.json')
       video_dir = os.path.join(args.dataset_dir,'videos')
    else:
       raise ValueError("dataset_name: longvideobench, videomme, mlvu")

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
                'video_info': video_data_infos[i]
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "videomme":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["videoID"]+'.mp4'),
                'video_info': video_data_infos[i]
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "mlvu":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["question_type"], video_data["video"]),
                'video_info': video_data_infos[i]
            } for i, video_data in enumerate(video_datas)
        ]

    save_dicts, all_time_logs = run_gw_on_gpus(all_tasks, args)
    total_time = sum(t["total_time"] for t in all_time_logs) / len(all_time_logs)
    args.logger.info(f"[INFO] Total time taken (gw): {total_time:.2f} seconds")
    
    save_lists = []
    for video_index, save_dict in save_dicts:
        save_lists.append(save_dict)

    with open(os.path.join(args.output_dir, f"{args.dataset_name}_clip_info.pkl"), 'wb') as f:
        pickle.dump(save_lists, f)
    args.info_dir = os.path.join(args.output_dir, f"{args.dataset_name}_clip_info.pkl")

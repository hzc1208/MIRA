import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Union, Optional, Dict, Literal
import decord
import cv2
from pathlib import Path
import os, sys, time, json, pickle, math
from zipfile import ZipFile
import traceback


def extract_batch_features(
    model,
    image_processor,
    tokenizer,
    images: Optional[List[Image.Image]] = None,
    texts: Optional[List[str]] = None,
    choice: Literal["image", "dense_image", "text"] = "image",
    batch_size: int = 100
) -> torch.Tensor:

    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'module'):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    all_features = []
    with torch.no_grad():
        if "image" in choice:
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                inputs = image_processor(images=batch, return_tensors='pt')['pixel_values'].to(device)
                if choice == "image":
                    feat = model.get_image_features(inputs).to(device)
                else:
                    feat = model.get_image_dense_features(inputs).to(device)
                feat = F.normalize(feat, p=2, dim=-1)
                all_features.append(feat.cpu())
        else:
            batch = torch.tensor(tokenizer(texts, max_length=248, padding="max_length", truncation=True).input_ids, dtype=torch.long).to(device)
            feat = model.get_text_features(batch, walk_short_pos=False).to(device)
            feat = F.normalize(feat, p=2, dim=-1)
            all_features.append(feat.cpu())
    return torch.cat(all_features, dim=0)


def compute_pairwise_similarity(
    img_features: torch.Tensor,
    text_features: Optional[torch.Tensor] = None,
    chunk_size: int = 100
) -> torch.Tensor:
    N = img_features.size(0)
    if text_features is None:
        sim_matrix = torch.zeros((N, N), dtype=torch.float32, device=img_features.device)
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            a = img_features[i:end_i]
            for j in range(0, N, chunk_size):
                end_j = min(j + chunk_size, N)
                b = img_features[j:end_j]
                sim_matrix[i:end_i, j:end_j] = torch.matmul(a, b.t())
        return sim_matrix
    else:
        sim_matrix = torch.zeros((N, text_features.size(0)), dtype=torch.float32, device=img_features.device)
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            a = img_features[i:end_i]
            if a.ndim == 2:
                sim_matrix[i:end_i] = torch.matmul(a, text_features.t())
            else:
                N_, patch_num = a.size(0), a.size(1)
                region_sim = torch.matmul(a.view(-1, a.size(-1)), text_features.t()).view(N_, patch_num, -1)
                sim_matrix[i:end_i] = region_sim.mean(dim=1)
        return sim_matrix


def split_video_scene_level1(
    sim_matrix: torch.Tensor,
    threshold_value: float = math.cos(math.radians(45)),
    time_window_size: int = 2
):
    index_to_video_scene, video_scene_to_index = {}, {}
    scene_ID, current_scene_begin_ID = 0, 0
    index_to_video_scene[0] = scene_ID
    N = sim_matrix.size(0)
    for i in range(1, N):
        if sim_matrix[i, current_scene_begin_ID:i].max().item() < threshold_value:
            if (i == N - 1) or \
                (sim_matrix[i+1:min(i+time_window_size+1, N), current_scene_begin_ID:i].max().item() < threshold_value):
                video_scene_to_index[scene_ID] = (current_scene_begin_ID, i - 1)
                scene_ID += 1
                index_to_video_scene[i] = scene_ID
                current_scene_begin_ID = i
            else:
                index_to_video_scene[i] = scene_ID
        else:
            index_to_video_scene[i] = scene_ID
    video_scene_to_index[scene_ID] = (current_scene_begin_ID, N - 1)

    return index_to_video_scene, video_scene_to_index


def split_video_scene_level2(
    sim_matrix: torch.Tensor,
    video_scene_to_index: Dict[int, Tuple[int, int]],
    avg_split_length: int = 4,
    threshold_value: float = math.cos(math.radians(15))
):
    index_to_subscene, subscene_to_index = {}, {}
    scene_ID = 0
    for video_scene_ID, (start_idx, end_idx) in video_scene_to_index.items():
        current_scene_begin_ID = start_idx
        if (end_idx - start_idx + 1) > avg_split_length:
            adjacent_sim_array = np.array([
                sim_matrix[i, i + 1] for i in range(start_idx, end_idx)
            ])
            split_num = int((end_idx - start_idx + 1) / avg_split_length)
            subscene_split_indices = np.argsort(adjacent_sim_array)[:split_num] + start_idx
            subscene_split_indices = set(subscene_split_indices)
            for i in range(start_idx, end_idx):
                if (i in subscene_split_indices) and (adjacent_sim_array[i - start_idx] < threshold_value):
                    subscene_to_index[scene_ID] = (current_scene_begin_ID, i)
                    index_to_subscene[i] = scene_ID
                    scene_ID += 1
                    current_scene_begin_ID = i + 1
                else:
                    index_to_subscene[i] = scene_ID
        else:
            for i in range(start_idx, end_idx):
                index_to_subscene[i] = scene_ID
        index_to_subscene[end_idx] = scene_ID
        subscene_to_index[scene_ID] = (current_scene_begin_ID, end_idx)
        scene_ID += 1

    return index_to_subscene, subscene_to_index


def load_video_or_frames(
    args,
    source: Union[str, Path],
    frame_num: int,
    grid_size: int = 224
) -> Tuple[List[Image.Image], List[int]]:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Path does not exist: {source}")
    frames: List[Image.Image] = []

    video_suffixes = [".mp4", ".avi", ".mov", ".mkv"]
    if source.is_file() and source.suffix.lower() in video_suffixes:
        vr = decord.VideoReader(str(source))
        total_frames = len(vr)
        fps = min(int(vr.get_avg_fps()), max(int(total_frames / frame_num), 1))
        frame_indices = [i*fps for i in range(int(total_frames / fps))]
        for idx in frame_indices:
            img = vr[idx].asnumpy()
            img = Image.fromarray(img).convert("RGB").resize((grid_size, grid_size))
            frames.append(img)
    elif source.is_dir():
        args.logger.info(f"[INFO] Loading frames from directory: {source}")
        img_files = []
        for root, _, files in os.walk(str(source)):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                    img_files.append(os.path.join(root, file))
        def extract_number(path):
            filename = Path(path).stem
            if filename.startswith("keyframe_"):
                return int(filename.split("keyframe_")[-1])
            return int(filename)
        img_files.sort(key=extract_number)
        frame_indices = []
        for i, img_path in enumerate(img_files):
            img = Image.open(img_path).convert("RGB").resize((grid_size, grid_size))
            frames.append(img)
            frame_indices.append(i)
        args.logger.info(f"[INFO] Loaded {len(frames)} images")
    elif source.suffix.lower() == ".zip":
        args.logger.info(f"[INFO] Loading frames from ZIP archive: {source}")
        img_files_in_zip = []
        with ZipFile(source, 'r') as zip_obj:
            for file in zip_obj.namelist():
                ext = Path(file).suffix.lower()
                if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                    img_files_in_zip.append(file)
        def extract_number_from_zip(path):
            filename = Path(path).stem
            if filename.startswith("keyframe_"):
                return int(filename.split("keyframe_")[-1])
            return int(filename)
        img_files_in_zip.sort(key=extract_number_from_zip)
        frame_indices = []
        with ZipFile(source, 'r') as zip_obj:
            for i, img_file in enumerate(img_files_in_zip):
                with zip_obj.open(img_file) as f:
                    img = Image.open(f).convert("RGB").resize((grid_size, grid_size))
                    frames.append(img)
                    frame_indices.append(i)
        args.logger.info(f"[INFO] Loaded {len(frames)} images from ZIP")
    else:
        raise ValueError("source must be a video file, image directory, or a .zip archive")
    return frames, frame_indices


def process_video(task, shared_model, args):
    try:
        sampling_time, scenes_time = 0, 0
        global_start_time = time.time()

        video_index, video_path, video_query = task['video_index'], task['video_path'], task['video_question']
        model, image_processor, tokenizer = shared_model
        # Step 1: Load frames
        start_time = time.time()
        frames_pil, frame_indices = load_video_or_frames(args, video_path, args.keyframe_num + args.causalframe_num)
        sampling_time += (time.time() - start_time)

        # Step 2.1: Extract features and build semantic-based scene segmentation
        img_features = extract_batch_features(
            model, image_processor, None, images=frames_pil, choice="image"
        )

        start_time = time.time()
        i2i_sim_matrix = compute_pairwise_similarity(img_features, None)
        index_to_video_scene, video_scene_to_index = split_video_scene_level1(i2i_sim_matrix)
        scenes_time += (time.time() - start_time)

        # Step 2.2: Calculate i2t similarity and filter irrelevant scenes
        txt_features = extract_batch_features(
            model, None, tokenizer, texts=[video_query], choice="text"
        )
        i2t_sim_matrix = compute_pairwise_similarity(img_features, txt_features).mean(dim=1).cpu().numpy()

        # Step 3.1: Extract dense image features and build fine-grained sub-scene segmentation
        img_dense_features = extract_batch_features(
            model,
            image_processor,
            None,
            images=frames_pil,
            choice="dense_image"
        )

        start_time = time.time()
        index_to_subscene, subscene_to_index = split_video_scene_level2(
            i2i_sim_matrix,
            video_scene_to_index
        )
        scenes_time += (time.time() - start_time)

        # Step 3.2: Calculate mixed i2t similarity
        i2t_sim_matrix_2 = 0.5 * i2t_sim_matrix +\
            0.5 * compute_pairwise_similarity(img_dense_features, txt_features).mean(dim=1).cpu().numpy()

        # Step 4: Return top K frames based on similarity
        top_k_relevant_indices = np.argsort(i2t_sim_matrix_2, axis=0)[::-1]
        res_frame_indices = []
        for idx in top_k_relevant_indices:
            res_frame_indices.append(frame_indices[idx])
            if len(res_frame_indices) >= args.keyframe_num:
                break

        args.logger.info(f"[INFO] Processing video {video_index}: {video_path} {len(frame_indices)} frames, "
            f"{len(video_scene_to_index)} scenes, {len(subscene_to_index)} subscenes.")

        return (video_index, sorted(res_frame_indices)), (video_index, \
            {
                "index_to_frame_idx": {i: frame_indices[i] for i in range(len(frame_indices))},
                "index_to_score": {i: i2t_sim_matrix_2[i] for i in range(len(frame_indices))},
                "index_to_subscene": index_to_subscene,
                "subscene_to_index": subscene_to_index
            }), \
            {
                "sampling_time": sampling_time,
                "scenes_time": scenes_time,
                "feature_time": time.time() - global_start_time - sampling_time - scenes_time
            }
    except Exception as e:
        args.logger.error(f"[ERROR] Failed to process video {video_index}: {video_path}")
        args.logger.error(f"[ERROR] Exception: {str(e)}")
        args.logger.error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        return None, None, None


def init_clip_model(model_root):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).to("cuda").eval()
    image_processor = AutoImageProcessor.from_pretrained(model_root)
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    return (model, image_processor, tokenizer)


def run_clip_on_gpu(my_tasks, args, gpu_idx):
    device_id = args.gpus[gpu_idx]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    results, save_dicts, time_logs = [], [], []
    # Filter tasks assigned to this GPU
    if not my_tasks:
        args.logger.info(f"[INFO] No tasks assigned to GPU {device_id}, skipping.")
        return
    args.logger.info(f"[INFO] Starting {len(my_tasks)} tasks on GPU {device_id}")

    # Load model once per GPU
    shared_model = init_clip_model(args.clip_root)
    for task in my_tasks:
        result = None
        while result is None:
            result, save_dict, save_times = process_video(task, shared_model, args)
        results.append(result)
        save_dicts.append(save_dict)
        time_logs.append(save_times)

    return results, save_dicts, time_logs


def run_clip_on_gpus(all_tasks, args):
    num_gpus = len(args.gpus)
    results, save_dicts, all_time_logs = [], [], []

    # Group tasks by GPU assignment
    gpu_task_groups = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        gpu_idx = i % num_gpus
        gpu_task_groups[gpu_idx].append(task)

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(run_clip_on_gpu, gpu_task_groups[gpu_idx], args, gpu_idx) for gpu_idx in range(num_gpus)]
        for future in futures:
            result, save_dict, time_logs = future.result()
            results.extend(result)
            save_dicts.extend(save_dict)
            all_time_logs.extend(time_logs)

    return sorted(results, key=lambda x: x[0]), sorted(save_dicts, key=lambda x: x[0]), all_time_logs


def run_clip_main(args):
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

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_name == "longvideobench":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["video_path"]),
                'video_question': video_data["question"] + ','.join(video_data["candidates"])
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "videomme":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["videoID"]+'.mp4'),
                'video_question': video_data["question"] + ','.join(video_data["options"])
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "mlvu":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["question_type"], video_data["video"]),
                'video_question': video_data["question"] + ','.join(video_data["candidates"])
            } for i, video_data in enumerate(video_datas)
        ]
    elif args.dataset_name == "egoschema":
        all_tasks = [
            {
                'video_index': i,
                'video_path': os.path.join(video_dir, video_data["video_idx"]+'.mp4'),
                'video_question': video_data["question"] + ','.join(video_data["option"])
            } for i, video_data in enumerate(video_datas)
        ]

    dataset_key_res, save_dicts, all_time_logs = run_clip_on_gpus(all_tasks, args)

    avg_sampling_time = sum(t["sampling_time"] for t in all_time_logs) / len(all_time_logs)
    avg_scenes_time = sum(t["scenes_time"] for t in all_time_logs) / len(all_time_logs)
    avg_feature_time = sum(t["feature_time"] for t in all_time_logs) / len(all_time_logs)

    args.logger.info(
        f"[INFO] Sampling = {avg_sampling_time:.2f} seconds, "
        f"Scenes = {avg_scenes_time:.2f} seconds, "
        f"Feature Extraction = {avg_feature_time:.2f} seconds"
    )

    dataset_key_fn, save_lists = [], []
    for video_index, res_frame_indices in dataset_key_res:
        dataset_key_fn.append(res_frame_indices)
    for video_index, save_dict in save_dicts:
        save_lists.append(save_dict)

    with open(os.path.join(args.output_dir, f"{args.dataset_name}_top{args.keyframe_num}_frames_clip.json"), 'w') as f:
        json.dump(dataset_key_fn, f)
    with open(os.path.join(args.output_dir, f"{args.dataset_name}_clip_info.pkl"), 'wb') as f:
        pickle.dump(save_lists, f)
    args.fn_dir = os.path.join(args.output_dir, f"{args.dataset_name}_top{args.keyframe_num}_frames_clip.json")
    args.info_dir = os.path.join(args.output_dir, f"{args.dataset_name}_clip_info.pkl")

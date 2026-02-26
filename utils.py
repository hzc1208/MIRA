import argparse
from PIL import Image, ImageDraw, ImageFont
from clip_model import load_video_or_frames
from video_pool import build_causal_inf_subscenes_dict
from typing import List, Dict, Union, Tuple, Optional, Literal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import os, json, pickle, math, random


def visualize_pool_correctness(array: np.ndarray, acc):
    if not isinstance(array, np.ndarray):
        raise TypeError("Input 'array' must be a NumPy ndarray.")
    if array.ndim != 2:
        raise ValueError("Input 'array' must be 2-dimensional.")
    rows, cols = array.shape
    if rows != cols + 1:
        raise ValueError("Input 'array' must have shape [N+1, N].")

    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['font.size'] = 16
    except Exception as e:
        print(f"Error setting Times New Roman font: {e}. Using default font.")
    
    fig, ax = plt.subplots(figsize=(10, 8))

    vmin, vmax = array.min(), array.max()
    if vmin == vmax:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax or 1.0)
    else:
        gamma = 0.3
        norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    im = ax.imshow(array, cmap='viridis', norm=norm, aspect='auto')
    
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels([f"{j}/{cols}" for j in range(cols, 0, -1)], fontsize=16)
    ax.set_xlabel("Correct Ratio", fontsize=20)
    
    y_labels = ["Total"] + [f"pool-{i}" for i in range(1, cols+1)]
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(y_labels, fontsize=16)
    ax.set_ylabel("Category", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    if cols <= 20: 
        for i in range(rows):
            for j in range(cols):
                normalized_val = norm(array[i, j])
                text_color = "w" if normalized_val < 0.5 else "k"
                text = ax.text(j, i, f'{array[i, j]}',
                            ha="center", va="center", color=text_color, fontsize=18)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Upper Acc: {acc*100:.1f}%', rotation=270, labelpad=25, fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    fig.tight_layout()

    output_pdf_path = os.path.join(args.output_dir, f'{args.dataset_name}_video_pool_info.pdf')
    try:
        plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving PDF file: {e}")
    finally:
        plt.close(fig)


def create_image_grid(
    frames: List[Image.Image],
    frame_ids: List[int],
    frame_type: Dict[int, str],
    img_save_path: str,
    x_grid_dim: Optional[int] = None,
    specified_frame_ids: Optional[List[int]] = None,
    grid_size: int = 224,
) -> Image.Image:
    assert len(frames) == len(frame_ids)
    if specified_frame_ids is not None:
        frames = [frame for frame, fid in zip(frames, frame_ids) if fid in specified_frame_ids]
        frame_ids = [fid for fid in frame_ids if fid in specified_frame_ids]

    n = len(frames)
    x_grid_dim = math.ceil(math.sqrt(n)) if x_grid_dim is None else x_grid_dim
    y_grid_dim = math.ceil(n / x_grid_dim)
    image_grid = Image.new('RGB', (x_grid_dim * grid_size, y_grid_dim * grid_size), color='white')
    draw = ImageDraw.Draw(image_grid)

    try:
        font = ImageFont.truetype("Times_New_Roman.ttf", size=round(32*(grid_size/224)))
    except:
        font = ImageFont.load_default().font_variant(size=round(32*(grid_size/224)))

    color_map = {
        "clip": "orange",
        "inf": "blue",
        "causal": "lightblue"
    }
    color_rgb = {
        "orange": (255, 165, 0),
        "blue": (0, 0, 255),
        "lightblue": (173, 216, 230)
    }

    for idx in range(n):
        row = idx // x_grid_dim
        col = idx % x_grid_dim
        x = col * grid_size
        y = row * grid_size

        current_frame_id = frame_ids[idx]
        current_type = frame_type.get(current_frame_id, "inf")
        border_color = color_rgb.get(color_map.get(current_type, "blue"), (0, 0, 255))
        text_bg_color = color_map.get(current_type, "blue")

        img = frames[idx].resize((grid_size, grid_size), Image.LANCZOS)
        bordered_img = Image.new("RGB", (grid_size, grid_size), color=(255, 255, 255))
        border_width = round(8 * (grid_size / 224))
        inner_img = img.crop((
            border_width, border_width,
            grid_size - border_width, grid_size - border_width
        ))
        bordered_img.paste(inner_img, (border_width, border_width))
        draw_borders = ImageDraw.Draw(bordered_img)
        draw_borders.rectangle(
            [(0, 0), (grid_size - 1, grid_size - 1)],
            outline=border_color,
            width=border_width
        )
        image_grid.paste(bordered_img, (x, y))

        text = str(frame_ids[idx])
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x + grid_size - text_width - border_width
        text_y = y

        box_coords = [
            (text_x, text_y + border_width),
            (text_x + text_width, text_y + text_height + border_width + 4)
        ]
        draw.rectangle(box_coords, fill=text_bg_color)

        draw.text((text_x, text_y), text, fill="white", font=font)

    image_grid.save(img_save_path, dpi=(450, 450), quality=95, subsampling=0)
    return image_grid


def _extract_all_answers(dataset_name: str, jsonl_paths: List[str]):
    all_answers: Dict[int, List[Tuple[Optional[str], Optional[str]]]] = {}
    question_types: Dict[int, str] = {}
    num_files = len(jsonl_paths)

    for file_index, file_path in enumerate(jsonl_paths):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            doc_id = data.get("doc_id")
                            if doc_id is not None:
                                if doc_id not in all_answers:
                                    all_answers[doc_id] = [(None, None)] * num_files
                                pred_answer = data.get("filtered_resps")[0].rstrip('.')
                                
                                if dataset_name == "longvideobench":
                                    true_answer = chr(int(data.get("target"))+ord('A'))
                                    question_types[doc_id] = data.get("doc", {}).get("duration_group", "unknown")
                                elif dataset_name == "videomme":
                                    true_answer = data.get("target")
                                    question_types[doc_id] = data.get("doc", {}).get("duration", "unknown")
                                elif dataset_name == "mlvu":
                                    true_answer = data.get("doc").get("answer_letter")
                                    question_types[doc_id] = data.get("doc", {}).get("question_type", "unknown")

                                all_answers[doc_id][file_index] = (pred_answer, true_answer) 

                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not decode JSON in file {file_path}: {e}")
                        except Exception as e:
                            print(f"Warning: An error occurred while processing a line in {file_path}: {e}")

        except FileNotFoundError:
            print(f"Error: File not found {file_path}")
        except Exception as e:
            print(f"Error: An unexpected error occurred while processing file {file_path}: {e}")

    return all_answers, question_types


def print_video_pools_acc_info(dataset_name: str, jsonl_paths: List[str]):
    all_answers_dict, question_types = _extract_all_answers(dataset_name, jsonl_paths)
    correct_sample_num, total_sample_num = dict(), dict()
    pool_correct_sample_num = [dict() for _ in range(len(jsonl_paths))]
    vis_matrix = np.zeros((len(jsonl_paths)+1, len(jsonl_paths)), dtype=int)
    for doc_id, answer_pairs_list in all_answers_dict.items():
        correct_doc_num = 0
        q_type = question_types.get(doc_id, "unknown")
        if q_type not in correct_sample_num:
            correct_sample_num[q_type] = 0
            total_sample_num[q_type] = 0
        total_sample_num[q_type] += 1
        for pred_answer, true_answer in answer_pairs_list:
            if (pred_answer == true_answer):
                correct_doc_num += 1
        for pool_id, (pred_answer, true_answer) in enumerate(answer_pairs_list):
            if q_type not in pool_correct_sample_num[pool_id]:
                pool_correct_sample_num[pool_id][q_type] = 0
            if pred_answer == true_answer:
                pool_correct_sample_num[pool_id][q_type] += 1
                vis_matrix[pool_id+1, -correct_doc_num] += 1
        if correct_doc_num > 0:
            vis_matrix[0, -correct_doc_num] += 1
            correct_sample_num[q_type] += 1

    if dataset_name == "mlvu":
        total_acc = sum([correct_sample_num[k] / total_sample_num[k] if total_sample_num[k] > 0 else 0 for k in correct_sample_num.keys()]) / len(correct_sample_num) if len(correct_sample_num) > 0 else 0
    else:
        total_acc = sum(correct_sample_num.values()) / sum(total_sample_num.values()) if sum(total_sample_num.values()) > 0 else 0

    if dataset_name == "mlvu":
        print(f"Total Acc. for Video Pools: {[f'pool-{i+1}: {sum([pool_correct_sample_num[i][k]/total_sample_num[k] for k in total_sample_num.keys()]) / len(total_sample_num) if len(total_sample_num) > 0 else 0:.3f}' for i in range(len(jsonl_paths))]}")
    else:
        print(f"Total Acc. for Video Pools: {[f'pool-{i+1}: {sum(pool_correct_sample_num[i].values()) / sum(total_sample_num.values()) if sum(total_sample_num.values()) > 0 else 0:.3f}' for i in range(len(jsonl_paths))]}")

    print(f"Detail Acc. for Video Pools by Question Type: {[f'pool-{i+1}-{k}: {pool_correct_sample_num[i][k] / total_sample_num[k] if total_sample_num[k] > 0 else 0:.3f}' for i in range(len(jsonl_paths)) for k in total_sample_num.keys()]}")

    visualize_pool_correctness(vis_matrix, total_acc)
    return total_acc


def visualize_selected_frames(args, all_tasks):
    with open(args.clip_infos_dir,'rb') as f:
        clip_infos = pickle.load(f)
    with open(args.causal_infos_dir,'rb') as f:
        causal_infos = pickle.load(f)
    load_fn_num = Path(args.clip_infos_dir).parent.name.split('_fn_')[1]
    load_fn_num = int(load_fn_num.split('_')[0]) + int(load_fn_num.split('_')[1])

    for video_idx, (clip_info, causal_info) in enumerate(zip(clip_infos, causal_infos)):
        if video_idx not in args.specified_video_id:
            continue

        frames_pil, _ = load_video_or_frames(all_tasks[video_idx]['video_path'], load_fn_num, grid_size=384)
        index_to_frame_idx = clip_info["index_to_frame_idx"]
        sorted_clip_indices = sorted(clip_info["index_to_score"], key=lambda k: clip_info["index_to_score"][k], reverse=True)
        # sorted_cap_indices = sorted(causal_info["index_to_cap_relevant_score"], key=lambda k: causal_info["index_to_cap_relevant_score"][k], reverse=True)
        sorted_causal_indices = sorted(causal_info["index_to_causal_score"], key=lambda k: causal_info["index_to_causal_score"][k], reverse=True)
        causal_inf_dict = build_causal_inf_subscenes_dict(clip_info["frames_for_cap_dict"], sorted_causal_indices)
        sorted_inf_indices = [idx for idx in sorted_clip_indices if idx in set(causal_inf_dict.keys())]
        # sorted_inf_indices = [idx for idx in sorted_cap_indices if idx in set(causal_inf_dict.keys())]

        for clip_fn_num, specified_frame_ids in zip(args.clip_fn_num, args.specified_frame_ids):
            top_k_clip_indices = sorted_clip_indices[:clip_fn_num]
            frame_type = {idx: "clip" for idx in top_k_clip_indices}
            top_k_clip_indices_set, top_k_causal_inf_indices, idx = set(top_k_clip_indices), [], 0

            while (len(top_k_causal_inf_indices) + len(top_k_clip_indices) < args.fn_num) and (idx < len(sorted_inf_indices)):
                inf_fn_idx = sorted_inf_indices[idx]
                if (inf_fn_idx not in top_k_clip_indices_set) and (inf_fn_idx not in top_k_causal_inf_indices):
                    top_k_causal_inf_indices.append(inf_fn_idx)
                    frame_type[inf_fn_idx] = "inf"
                if len(top_k_causal_inf_indices) + len(top_k_clip_indices) < args.fn_num:
                    for causal_fn_idx in causal_inf_dict.get(inf_fn_idx, []):
                        if (causal_fn_idx not in top_k_clip_indices_set) and (causal_fn_idx not in top_k_causal_inf_indices):
                            top_k_causal_inf_indices.append(causal_fn_idx)
                            frame_type[causal_fn_idx] = "causal"
                            break
                idx += 1

            if len(top_k_causal_inf_indices) + len(top_k_clip_indices) < args.fn_num:
                remain_fn_num = args.fn_num - len(top_k_causal_inf_indices) - len(top_k_clip_indices)
                top_k_causal_inf_indices_set = set(top_k_causal_inf_indices)
                for fn_idx in sorted_clip_indices[clip_fn_num:]:
                    if remain_fn_num <= 0:
                        break
                    if fn_idx not in top_k_causal_inf_indices_set:
                        top_k_causal_inf_indices.append(fn_idx)
                        frame_type[fn_idx] = "clip"
                        remain_fn_num -= 1

            selected_frames = sorted(top_k_clip_indices + top_k_causal_inf_indices)
            frame_type = {index_to_frame_idx[i]: frame_type[i] for i in selected_frames}
            img_save_path = os.path.join(args.output_dir, f"video_{video_idx}_selected_frames_{clip_fn_num}_{args.fn_num-clip_fn_num}.jpg")
            create_image_grid([frames_pil[i] for i in selected_frames],
                            [index_to_frame_idx[i] for i in selected_frames],
                            frame_type, img_save_path,
                            x_grid_dim=len(specified_frame_ids)//2 if specified_frame_ids != [-1] else None,
                            specified_frame_ids=specified_frame_ids if specified_frame_ids != [-1] else None,
                            grid_size=384)


def parse_json_list(s):
    try:
        result = json.loads(s)
        if not isinstance(result, list) or not all(isinstance(x, list) for x in result):
            raise ValueError
        return result
    except (json.JSONDecodeError, ValueError):
        raise argparse.ArgumentTypeError("Argument must be a valid JSON list of lists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualization for Selected LongVideo Frames')
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing videos or image folders")
    parser.add_argument("--clip_infos_dir", type=str, default=None, help="Path to the clip info pickle file")
    parser.add_argument("--causal_infos_dir", type=str, default=None, help="Path to the causal info pickle file")
    parser.add_argument("--fn_num", type=int, default=64, help="Total video frame capacity")
    parser.add_argument("--clip_fn_num", nargs='+', type=int, default=[64, 48, 32, 16, 0], help="Clip-based video frame capacity")
    parser.add_argument("--specified_video_id", nargs='+', type=int, default=[], help="Specific video ID to visualize")
    parser.add_argument("--specified_frame_ids", type=parse_json_list, default=[[-1]], help="List of specific frame IDs to visualize")
    parser.add_argument("--visualize", action='store_true', help="Whether to visualize selected frames for each video")

    parser.add_argument("--response_jsonl_paths", nargs='+', type=str, default=[], help="List of JSONL files containing model responses for different video pools")
    args = parser.parse_args()

    args.dataset_name = Path(args.clip_infos_dir).parent.name.split('_fn_')[0]
    args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset_name + f"_visualization")
    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.response_jsonl_paths) > 0:
        print(f"Upperbound Acc. of Video Pools: {print_video_pools_acc_info(args.dataset_name, args.response_jsonl_paths):.3f}")

    if args.visualize:
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

        if args.dataset_name == "longvideobench":
            all_tasks = [
                {
                    'video_index': i,
                    'video_path': os.path.join(video_dir, video_data["video_path"])
                } for i, video_data in enumerate(video_datas)
            ]
        elif args.dataset_name == "videomme":
            all_tasks = [
                {
                    'video_index': i,
                    'video_path': os.path.join(video_dir, video_data["videoID"]+'.mp4')
                } for i, video_data in enumerate(video_datas)
            ]
        elif args.dataset_name == "mlvu":
            all_tasks = [
                {             
                    'video_index': i,
                    'video_path': os.path.join(video_dir, video_data["question_type"], video_data["video"])
                } for i, video_data in enumerate(video_datas)
            ]
        
        visualize_selected_frames(args, all_tasks)

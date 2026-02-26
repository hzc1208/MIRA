import argparse
import json, pickle, os, time

def build_causal_inf_subscenes_dict(fn_window: dict, sorted_causal_indices: list):
    causal_inf_dict = dict()
    for inf_fn, neigh_fns in fn_window.items():
        causal_inf_dict[inf_fn] = [idx for idx in sorted_causal_indices if idx in set(neigh_fns)]
    return causal_inf_dict


def build_video_frame_pool(args):
    global_start_time = time.time()

    with open(args.clip_infos_dir,'rb') as f:
        clip_infos = pickle.load(f)
    with open(args.causal_infos_dir,'rb') as f:
        causal_infos = pickle.load(f)

    final_selected_frames = {clip_fn_num: [] for clip_fn_num in args.clip_fn_num}
    for clip_info, causal_info in zip(clip_infos, causal_infos):
        index_to_frame_idx = clip_info["index_to_frame_idx"]
        sorted_clip_indices = sorted(clip_info["index_to_score"], key=lambda k: clip_info["index_to_score"][k], reverse=True)
        # sorted_cap_indices = sorted(causal_info["index_to_cap_relevant_score"], key=lambda k: causal_info["index_to_cap_relevant_score"][k], reverse=True)
        sorted_causal_indices = sorted(causal_info["index_to_causal_score"], key=lambda k: causal_info["index_to_causal_score"][k], reverse=True)
        causal_inf_dict = build_causal_inf_subscenes_dict(clip_info["frames_for_cap_dict"], sorted_causal_indices)
        sorted_inf_indices = [idx for idx in sorted_clip_indices if idx in set(causal_inf_dict.keys())]

        # sorted_inf_indices = [idx for idx in sorted_causal_indices]
        # sorted_inf_indices = [idx for idx in sorted_cap_indices if idx in set(causal_inf_dict.keys())]

        for clip_fn_num in args.clip_fn_num:
            top_k_clip_indices = sorted_clip_indices[:clip_fn_num]
            top_k_clip_indices_set, top_k_causal_inf_indices, idx = set(top_k_clip_indices), [], 0

            while (len(top_k_causal_inf_indices) + len(top_k_clip_indices) < args.fn_num) and (idx < len(sorted_inf_indices)):
                inf_fn_idx = sorted_inf_indices[idx]
                if (inf_fn_idx not in top_k_clip_indices_set) and (inf_fn_idx not in top_k_causal_inf_indices):
                    top_k_causal_inf_indices.append(inf_fn_idx)
                if len(top_k_causal_inf_indices) + len(top_k_clip_indices) < args.fn_num:
                    remain_causal_num = 2
                    for causal_fn_idx in causal_inf_dict.get(inf_fn_idx, []):
                        if (causal_fn_idx not in top_k_clip_indices_set) and (causal_fn_idx not in top_k_causal_inf_indices):
                            top_k_causal_inf_indices.append(causal_fn_idx)
                            # break
                            remain_causal_num -= 1
                            if remain_causal_num == 0:
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
                        remain_fn_num -= 1

            final_selected_frames[clip_fn_num].append(sorted([index_to_frame_idx[i] for i in top_k_clip_indices] + [index_to_frame_idx[i] for i in top_k_causal_inf_indices]))

    os.makedirs(args.output_dir, exist_ok=True)
    for clip_fn_num in args.clip_fn_num:
        with open(os.path.join(args.output_dir, f"selected_frames_{clip_fn_num}_{args.fn_num-clip_fn_num}.json"), 'w') as f:
            json.dump(final_selected_frames[clip_fn_num], f)

    total_time = (time.time() - global_start_time) / len(clip_infos)
    print(f"[INFO] Total time taken: {total_time:.6f} seconds")
    print(len(clip_infos))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Building Pools for LongVideo Frames')
    parser.add_argument("--clip_infos_dir", type=str, required=True, help="Directory containing clip_infos file")
    parser.add_argument("--causal_infos_dir", type=str, required=True, help="Directory containing causal_infos file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file")
    parser.add_argument("--fn_num", type=int, default=64, help="Total video frame capacity")
    parser.add_argument("--clip_fn_num", nargs='+', type=int, default=[0, 8, 16, 24, 32, 40, 48, 56, 64], help="Clip-based video frame capacity")
    args = parser.parse_args()
    
    build_video_frame_pool(args=args)

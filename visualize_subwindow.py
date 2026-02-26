import argparse
from video_pool import build_causal_inf_subscenes_dict
import pickle, random, sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def calculate_clip_causal_ranking_info(clip_info, causal_info, neigh_fn_idx="all"):
    sorted_clip_indices = sorted(clip_info["index_to_score"], key=lambda k: clip_info["index_to_score"][k], reverse=True)
    sorted_causal_indices = sorted(causal_info["index_to_causal_score"], key=lambda k: causal_info["index_to_causal_score"][k], reverse=True)
    fn_windows = clip_info["frames_for_cap_dict"]
    clip_ranking = {fn_idx: i+1 for i, fn_idx in enumerate(sorted_clip_indices)}
    causal_ranking = {fn_idx: i+1 for i, fn_idx in enumerate(sorted_causal_indices)}
    meta_info = {"avg_dist": [], "neigh_clip_ranking": [], "neigh_causal_ranking": [], "anchor_clip_ranking": []}
    causal_inf_dict = build_causal_inf_subscenes_dict(fn_windows, sorted_causal_indices)

    try:
        for inf_fn, neigh_fns in fn_windows.items():
            meta_info["anchor_clip_ranking"].append(clip_ranking[inf_fn])
            if neigh_fn_idx == "all":
                for neigh_fn in neigh_fns:
                    meta_info["neigh_clip_ranking"].append(clip_ranking[neigh_fn])
                    meta_info["neigh_causal_ranking"].append(causal_ranking[neigh_fn])
                    meta_info["avg_dist"].append(abs(inf_fn - neigh_fn))
            else:
                neigh_fn = causal_inf_dict[inf_fn][neigh_fn_idx]
                meta_info["neigh_clip_ranking"].append(clip_ranking[neigh_fn])
                meta_info["neigh_causal_ranking"].append(causal_ranking[neigh_fn])
                meta_info["avg_dist"].append(abs(inf_fn - neigh_fn))                
        
        meta_info["avg_dist"] = sum(meta_info["avg_dist"]) / len(meta_info["avg_dist"])
        meta_info["neigh_clip_ranking"] = sum(meta_info["neigh_clip_ranking"]) / len(meta_info["neigh_clip_ranking"])
        meta_info["neigh_causal_ranking"] = sum(meta_info["neigh_causal_ranking"]) / len(meta_info["neigh_causal_ranking"])
        meta_info["anchor_clip_ranking"] = sum(meta_info["anchor_clip_ranking"]) / len(meta_info["anchor_clip_ranking"])
        return meta_info
    except:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizing Subscene Windows')
    parser.add_argument("--clip_infos_dir", type=str, required=True, help="Directory containing clip_infos file")
    parser.add_argument("--causal_infos_dir", type=str, required=True, help="Directory containing causal_infos file")
    args = parser.parse_args()

    with open(args.clip_infos_dir,'rb') as f:
        clip_infos = pickle.load(f)
    with open(args.causal_infos_dir,'rb') as f:
        causal_infos = pickle.load(f)
    
    run_choices = ["all", 0]
    for neigh_fn_idx in run_choices:
        avg_meta_info = {"avg_dist": [], "neigh_clip_ranking": [], "neigh_causal_ranking": [], "anchor_clip_ranking": [], "avg_causal_fn_rate": []}
        for clip_info, causal_info in zip(clip_infos, causal_infos):
            meta_info = calculate_clip_causal_ranking_info(clip_info, causal_info, neigh_fn_idx)
            if meta_info is not None:
                avg_meta_info["avg_dist"].append(meta_info["avg_dist"])
                avg_meta_info["neigh_clip_ranking"].append(meta_info["neigh_clip_ranking"])
                avg_meta_info["neigh_causal_ranking"].append(meta_info["neigh_causal_ranking"])
                avg_meta_info["anchor_clip_ranking"].append(meta_info["anchor_clip_ranking"])
                causal_fns = set()
                for causal_fn in causal_info["index_to_causal_score"].keys():
                    causal_fns.add(causal_fn)
                causal_fn_rate = len(causal_fns) / len(clip_info["index_to_score"].keys())
                avg_meta_info["avg_causal_fn_rate"].append(causal_fn_rate)
        
        avg_meta_info["avg_dist"] = sum(avg_meta_info["avg_dist"]) / len(avg_meta_info["avg_dist"])
        avg_meta_info["neigh_clip_ranking"] = sum(avg_meta_info["neigh_clip_ranking"]) / len(avg_meta_info["neigh_clip_ranking"])
        avg_meta_info["neigh_causal_ranking"] = sum(avg_meta_info["neigh_causal_ranking"]) / len(avg_meta_info["neigh_causal_ranking"])
        avg_meta_info["anchor_clip_ranking"] = sum(avg_meta_info["anchor_clip_ranking"]) / len(avg_meta_info["anchor_clip_ranking"])
        avg_meta_info["avg_causal_fn_rate"] = sum(avg_meta_info["avg_causal_fn_rate"]) / len(avg_meta_info["avg_causal_fn_rate"])

        logging.info(f"neigh_fn_idx: {neigh_fn_idx} --> {avg_meta_info}")

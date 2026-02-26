import argparse
import logging, os, sys
from datetime import datetime
from clip_model import run_clip_main
from generate_subwindow import run_gw_main
from cap_model import run_cap_main
from causal_model import run_causal_main


def setup_logger(log_dir):
    log_dir = log_dir.rstrip("/\\")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding='utf-8')]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LongVideo Understanding')
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing videos or image folders")
    parser.add_argument("--keyframe_num", type=int, default=64, help="Number of keyframes to extract from each video")
    parser.add_argument("--causalframe_num", type=int, default=64, help="Number of causalframes to extract from each video")
    parser.add_argument("--clip_root", type=str, default="qihoo360/fg-clip-large",
                        choices=[
                            "qihoo360/fg-clip-base",
                            "qihoo360/fg-clip-large"
                        ],
                        help="FG-CLIP model used for feature extraction.")
    parser.add_argument("--vllm_root", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="VLLM model used for video scene caption.")
    parser.add_argument("--llm_root", type=str, default="Qwen/Qwen3-8B",
                        choices=[
                            "Qwen/Qwen3-1.7B",
                            "Qwen/Qwen3-8B"
                        ],
                        help="LLM model used for causal inference.")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0, 1, 2, 3], help="List of GPUs to use")
    parser.add_argument("--use_gw", action='store_true', help="Whether to generate global subwindows")
    args = parser.parse_args()

    args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"MIRA_{args.dataset_name}_fn_{args.keyframe_num}_{args.causalframe_num}")
    args.logger = setup_logger(args.output_dir)

    run_clip_main(args)
    args.info_dir = os.path.join(args.output_dir, f"{args.dataset_name}_clip_info.pkl")
    if args.use_gw:
        run_gw_main(args)
    run_cap_main(args)
    run_causal_main(args)

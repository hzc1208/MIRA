# MIRA: Multi-view Information Retrieval with Adaptive Routing for Test-time Long-video Comprehension
Code implementation for [MIRA: Multi-view Information Retrieval with Adaptive Routing for Test-time Long-video Comprehension](https://openreview.net/forum?id=LZb2kzO8tu) (*TMLR 2026*).

## 👨‍💻 Quick Usage
```
python3 main.py --dataset_name longvideobench --dataset_dir /home/to/datasets

python3 video_pool.py --clip_infos_dir longvideobench_fn_64_64/longvideobench_clip_info.pkl --causal_infos_dir longvideobench_fn_64_64/longvideobench_causal_frames.pkl --output_dir lvb_pool_fn64
```

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝:

```bibtex
@article{hao2026mira,
  title={MIRA: Multi-view Information Retrieval with Adaptive Routing for Test-time Long-video Comprehension},
  author={Hao, Zecheng and Ma, Wenxuan and Cui, Yufeng and Li, Shuang and Wang, Xinlong and Huang, Tiejun},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
# Online Convex Optimization with Unbounded Memory
This repository contains code for the paper "[Online Convex Optimization with Unbounded Memory](https://arxiv.org/abs/2210.09903)" published in NeurIPS 2023.

## Usage

1. Change directory to `online-linear-control`.
2. Run `python main.py --d {dimension} --rho {diagonal values of F} --upper_triangular_val {upper triangular value of F} --T {number of rounds}`.
3. The plots are stored in the `plots` directory.

## Citation

If you use this code, please cite the following paper
```
@inproceedings{kumar2023online,
  author    = "Kumar, Raunak and Dean, Sarah and Kleinberg, Robert ",
  title     = "Online Convex Optimization with Unbounded Memory",
  year      = "2023",
  booktitle = "Proceedings of the 37th Conference on Neural Information Processing Systems (\textbf{NeurIPS})"
}
```

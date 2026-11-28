# DPS 拓展方法

| 目录 | 作用位置 | 默认使用 |
| --- | --- | --- |
| [多种子温度场平均](temperature_ensemble_mean/README.md) | 采样完成后逐像素平均 | 四种子 |
| [区块温度投影](block_temperature_projection/README.md) | 末端恢复粗观测的块均温 | 是 |
| [粗温度插值参照](interpolated_temperature_baseline/README.md) | 独立参照，也是残差收缩的底图 | 可选 |
| [块内细节等比例收缩](uniform_detail_shrinkage/README.md) | 统一控制残差幅度 | 可选 |
| [块内细节分尺度收缩](multiscale_detail_shrinkage/README.md) | 分别控制三种尺度的残差幅度 | 可选 |
| [粗温度条件分支](coarse_temperature_condition_branch/README.md) | 去噪网络特征注入 | 历史基座选件 |
| [相对幅度引导](relative_amplitude_guidance/README.md) | 每个采样步调整观测修正量 | 研究对照 |

各方法的说明依据原代码和项目 `文本版` 中的论文逐项核对。`geometry.py` 提供无状态的整数分区边界和块平均。后处理接受数组。分尺度收缩通过参数接收投影函数。`../reconstruct_temperature.py` 负责组合调用。

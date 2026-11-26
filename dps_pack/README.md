# DPS 火焰温度场重建与傅里叶评价

**精简打包版 · 2026-11-26**

输入为 `16×21` 个非重叠区块的平均温度（K），输出为 `192×256` 的二维温度场（K）。主模型由多工况自编码器（AE）和潜空间 DDPM 先验组成。原生入口生成单种子温度场。默认组合对同一观测运行四条固定强度 DPS 轨迹，逐像素平均后投影到观测块均温，复用 L5 的 1000 步、ζ=1 配方。

## 包内容

| 位置 | 作用 |
| --- | --- |
| [flame_reconstruction/](flame_reconstruction/README.md) | 单种子原生 DPS、两阶段训练入口、模型结构和 L5 新基座权重 |
| [flame_reconstruction/native_dps/](flame_reconstruction/native_dps/README.md) | 区块均温观测、原训练日程和外部 Azula 采样器接口 |
| [dps_extensions/](dps_extensions/README.md) | 七个可分开使用的方法。各方法附原理、参数和论文关联 |
| [fourier_eval_plus/](fourier_eval_plus/README.md) | 可单独安装的二维场方向谱、功率误差和 Parseval 评价 |
| [reconstruct_temperature.py](reconstruct_temperature.py) | 组合入口，负责选择采样器、成员种子和后处理 |
| [reconstruction_recipe.json](reconstruction_recipe.json) | 固定 DPS 配方、四个通用种子，以及 L5 各正式帧的原始种子 |
| [DEPENDENCIES.md](DEPENDENCIES.md) | 依赖、离线安装方式与本地验证环境 |

参数保留 L5 新基座的 AE 和先验，以及 A600 分支与其历史配套 AE、先验。两个先验 checkpoint 均已提取为去噪网络张量。配置记录源文件和包内文件的 SHA-256。交付范围为源码、最终参数、必要配置及使用说明。

## 从粗观测运行

从这份 README 所在目录执行。`obs_K.npy` 需是形状 `(16,21)` 的实数数组。每行和每列按分区顺序排列。`--device auto` 在可用时选 CUDA，否则选 CPU。下列输出路径需尚未存在。

~~~powershell
# 原生固定强度 DPS，单种子，默认 1000 步
python -m flame_reconstruction.inference --observation obs_K.npy --out native_K.npy --seed 0

# 四种子温度场平均 + 区块温度投影
python reconstruct_temperature.py --observation obs_K.npy --out four_seed_K.npy --seeds 0 1 2 3

# 使用该 L5 正式帧的原始四种子。须自行提供这帧对应的粗观测。
python reconstruct_temperature.py --observation obs_K.npy --out l5_K.npy --l5-frame T1030-a0-f50

# 粗观测线性插值 + 块均对齐，只需要 NumPy
python reconstruct_temperature.py --observation obs_K.npy --out interpolation_K.npy --postprocess interpolation
~~~

`--sampler relative` 使用 L5 标定 β，可用 `--beta` 显式覆盖。`--sampler condition-branch` 使用 A600 及其配套历史模型。`--postprocess uniform` 和 `multiscale` 分别使用两种收缩。`--postprocess mean` 只计算平均。支持 `--steps` 调整采样步数；固定引导可通过 `--zeta` 调整。`--seeds` 接受一个或多个互异整数。首次检查可先用 `--steps 4`。

训练和外部数据接口见 [主程序说明](flame_reconstruction/README.md)。给定同网格的参考场 `reference_K.npy` 后，可直接运行频谱评价。

~~~powershell
python -m fourier_eval_plus --truth reference_K.npy --method dps=four_seed_K.npy --unit K --out spectrum --no-figures
~~~

验证结果见 [DEPENDENCIES.md](DEPENDENCIES.md#交付核验)。

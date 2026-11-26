# 火焰温度重建主程序

模型对 192×256 的单通道温度场做归一化，自编码器将场压到 48×64 潜变量。卷积和 Transformer 组成的去噪网络在潜空间预测噪声。反推时从标准高斯潜变量开始，原生 DPS 在每步把预测的干净潜变量解码并求块均值，再把观测差反传到当前潜变量。

- `inference.py` 从 `16×21` 块均温度直接推理输出 K，见[组合与推理用法](../README.md)。
- `runtime.py` 按 [model_config.json](model_config.json) 校验并加载两套网络的参数，保留原 DDPM 日程。
- `train.py` 用调用方提供的 K 单位 `(N,192,256)` 数组训练 AE，随后冻结 AE 并训练扩散噪声预测。采用原 AE 重建 MSE、先验噪声 MSE、Adam 学习率 0.0001 / 0.00001 和梯度范数裁剪 1。
- 模型代码主要来自 `local_work/flamediff_native_dps12_downsample`。训练入口保留两阶段目标和参数训练逻辑，简化为外部数组输入和最终参数输出。

## 用外部数据训练

AE 和先验须共享相同的温度归一化范围。示例中的工作数据、输出目录和训练步数由调用方提供。先验阶段写出可直接加载的 `model_config.json`。

~~~powershell
python -m flame_reconstruction.train ae --data train_K.npy --out ae_fit --min-k 304.6341857910156 --max-k 1630.6864013671875 --steps 33121 --batch-size 4
python -m flame_reconstruction.train prior --data train_K.npy --out prior_fit --ae ae_fit/autoencoder.pt --min-k 304.6341857910156 --max-k 1630.6864013671875 --steps 66094 --batch-size 4
python -m flame_reconstruction.inference --model-config prior_fit/model_config.json --observation obs_K.npy --out trained_prediction_K.npy
~~~

# 依赖与运行环境

| 功能 | 依赖 |
| --- | --- |
| 原生 DPS 与 A600 推理 | Python 3.12（实测）、PyTorch、与其匹配的 torchvision、NumPy ≥ 1.24、diffusers 0.32.2、Azula 0.11.1 |
| AE 与扩散先验两阶段训练 | PyTorch、NumPy、diffusers 0.32.2 |
| 独立场数组后处理 | NumPy ≥ 1.24，使用分尺度收缩时另需 SciPy |
| 独立傅里叶评价 | Python ≥ 3.10，NumPy ≥ 1.24。生成图像另需 Matplotlib ≥ 3.7 和 Pillow ≥ 9 |

历史运行和本地检查的主要版本为 Python 3.12.13、Torch 2.11.0+cu128、NumPy 2.5.2、diffusers 0.32.2、SciPy 1.18.1，GPU 为 RTX 4060 Laptop。普通推理支持 CPU，长链推荐使用 CUDA。自备训练数据和粗观测使用 NumPy 数值数组。

## 使用本机现有环境

在项目原位置打开本包，已有的 `pytorch` 环境可配合项目留存的 wheel 离线使用。前两行把 wheel 安装在系统临时目录，第三行让当前 PowerShell 会话引用该依赖。复制本包到别处后，可改成实际 wheel 的路径，或直接使用已安装同版本 Azula 的环境。

~~~powershell
$azulaDep = Join-Path $env:TEMP 'dps_pack_azula_0_11_1'
& 'D:\Anaconda\envs\pytorch\python.exe' -m pip install --no-index --no-deps --target $azulaDep '..\local_work\azula_l0\20261025_T8x\wheels\azula-0.11.1-py3-none-any.whl'
$env:PYTHONPATH = $azulaDep
& 'D:\Anaconda\envs\pytorch\python.exe' -m flame_reconstruction.inference --help
~~~

装在其他机器时，先选适合该机器的 PyTorch 与 torchvision，然后用 `python -m pip install -r requirements.txt` 安装重建依赖。Azula 安装器会处理其传递依赖。Azula 版本、采样类和源码指纹见 [upstream.json](flame_reconstruction/native_dps/upstream.json)。采样按原日程与种子驱动，跨设备或跨版本的浮点输出可能有差异。傅里叶工具也可单独安装，见 [工具说明](fourier_eval_plus/README.md)。

## 交付核验

验证在临时副本上执行。Azula 安装自本地 wheel，核对版本及两份采样源码的 SHA-256。GPU 上用同一合成观测、同一种子对照原实现。可用的入口、权重和验证等级见下表。

| 检查 | 结果 |
| --- | --- |
| 权重与日程 | 五个源文件的 SHA-256 与原配置一致。两份先验提取后的张量与原 checkpoint 逐项相等。AE 解码与先验噪声预测输出一致，日程与原 `DDPMScheduler()` 配置一致。 |
| 重建与可选采样器 | 原生 DPS、A600 和相对幅度引导在同一 GPU、同一输入与种子下的四步输出与原实现逐点一致。原生入口移开全部选件后仍通过。A600 拒绝不配套身份。A600 与相对幅度引导的两步 CPU 命令通过。 |
| 场后处理 | 四成员平均、区块投影及两种收缩与原实现的绝对差小于或等于 1e-10 K。四成员组合与独立插值 CLI 的逐块均温通过 1e-9 K 核验。 |
| 训练入口 | 使用临时温度数组分别完成 AE、先验、A600 各一次更新，成功加载新训练入口生成的 AE 与先验并推理。 |
| 傅里叶工具 | 离线独立安装成功，正弦场谱峰与 Parseval 空间 MSE 守恒通过。Hann 窗和无窗两种 CLI 均通过完整 FFT 复核，绘图生成 10 张图。包内修正了含 `#` 路径的读取，以及参考谱接近舍入误差时的比值复核。 |

推理核验使用 2–4 步的合成观测。完整 1000 步 L5 正式实验与长时间训练未在本次打包中重跑。

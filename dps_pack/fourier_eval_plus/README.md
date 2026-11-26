# 二维标量场傅里叶评价

对同一网格上的参考场和重建场计算空间误差、行列两个方向的单边功率谱、分频段谱偏差和配对误差。输入可为温度、速度分量或任意实值二维标量。接口支持单场 `(H,W)` 或同形状批量 `(N,H,W)`。

| 文件 | 内容 |
| --- | --- |
| `directional_spectrum.py` | 通用方向谱、频带与谱偏差 |
| `engine.py`、`contracts.py` | 数值 API 和计算调度 |
| [methods/](methods/README.md) | 论文功率谱 NRMSE 与 Parseval 频段误差 |
| `field_loaders.py`、`datasets.py`、`dps_inputs.py` | 数值文件、配对描述符和可选的历史 DPS 数值格式 |
| `runner.py`、`figures.py`、`report.py`、`verify.py` | CLI、绘图、报告与独立 FFT 复算 |

从 `dps_pack` 运行可直接使用源码。要单独部署，可把本目录复制到其他位置，用 `python -m pip install ./fourier_eval_plus` 安装。绘图额外依赖可用 `python -m pip install "./fourier_eval_plus[plot]"` 安装。

~~~powershell
python -m fourier_eval_plus --truth ref.npy --method dps=recon.npy --unit K --out evaluation --no-figures
~~~

安装后可用同参数的 `fourier-eval` 命令。`fourier-eval-verify --run evaluation` 对 CLI 输出做全 FFT 复核。删除 `--no-figures` 即生成图像。可重复指定 `--method 名称=文件` 比较多个模型。NPZ 字段写为 `文件.npz#字段名`。

~~~python
from fourier_eval_plus import evaluate

result = evaluate(reference_field, {"dps": reconstructed_field})
spatial_and_power = result.baseline
spectral_error = result.enhancements["paper27"]
paired_error = result.enhancements["parseval"]
~~~

详情见 [MATH.md](MATH.md)。由原 `fourier_eval_plus` 按文件提取，数值约定和默认配置沿用原工具。

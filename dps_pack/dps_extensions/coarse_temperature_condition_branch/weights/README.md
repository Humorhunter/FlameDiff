# A600 历史配套权重

| 文件 | 内容 |
| --- | --- |
| `condition_branch.pt` | 原 A600 分支权重与训练身份 |
| `autoencoder.pt` | 历史整合配置的 AE |
| `diffusion_prior.pt` | 历史整合配置第 65,160 步的去噪网络张量 |

这些文件来自 `local_work/dps12_integrated_candidate_20261120/weights`。原先验 checkpoint 中的 AE 与独立 AE 张量逐项相等，先验导出只保留去噪网络参数。源文件和包内指纹见 [model_config.json](../model_config.json)。这套先验对应 A600 的训练身份，供历史配方使用。

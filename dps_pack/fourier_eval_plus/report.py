"""Human-readable report without selecting or merging enhancement scores."""
def _value(stats, name='median'):
    v=stats[name]
    return 'n/a' if v is None else f'{v:.6g}'


def write_report(evaluation, path, unit, seconds):
    base=evaluation.baseline
    lines=['# 傅里叶评价增强版v2','',
           f'帧数：{len(base.frame_ids)}；单位：{unit}；核心计算：{seconds:.3f}秒。',
           '', '## 基础版对照（逐帧中位数）','',
           '| 方法 | MAE | RMSE | D | D_log |','| --- | ---: | ---: | ---: | ---: |']
    for label,row in base.summary['methods'].items():
        lines.append(f"| {label} | {_value(row['spatial']['MAE'])} | {_value(row['spatial']['RMSE'])} | {_value(row['overall']['D_overall'])} | {_value(row['overall']['D_log_overall'])} |")
    if 'paper27' in evaluation.enhancements:
        result=evaluation.enhancements['paper27']
        lines+=['','## paper27：公式(11)的方向联合谱NRMSE','',
                '沿用基础版的窗与去均值；合并两方向的平方和后开方，不平均两个相对误差。resolved/unresolved仅是名义采样频率划分。','',
                '| 方法 | 全非DC | 名义可分辨频段 | 名义未分辨频段 |','| --- | ---: | ---: | ---: |']
        for label,row in result['summary'].items():
            joint=row['joint']
            lines.append(f"| {label} | {_value(joint['all']['NRMSE_sp'])} | {_value(joint['resolved']['NRMSE_sp'])} | {_value(joint['unresolved']['NRMSE_sp'])} |")
    if 'parseval' in evaluation.enhancements:
        result=evaluation.enhancements['parseval']
        lines+=['','## parseval：完整空间MSE的分尺度归因','',
                '不加窗、不去均值。DC＋各频段MSE之和等于空间MSE；每个方向都是一次独立分解，不能相加。以下取逐帧均值，便于保持加法恒等式；异常帧均保留。','',
                f'| 方法 | 方向 | 频段 | 总误差/{unit}² | 幅度项/{unit}² | 幅度加权相位项/{unit}² |',
                '| --- | --- | --- | ---: | ---: | ---: |']
        for label,row in result['summary'].items():
            for axis,data in row['axes'].items():
                for band,block in data['bands'].items():
                    lines.append(f"| {label} | {axis} | {band} | {_value(block['error_mse'],'mean')} | {_value(block['magnitude_mse'],'mean')} | {_value(block['phase_mse'],'mean')} |")
    lines+=['','## 解释边界','',
            '- paper27比较功率分布，不保证位置正确；parseval比较配对误差，没有把相位单独归一化成评分。',
            '- 幅度加权相位项不是纯相位误差：预测幅度为零时相位项为零，缺失信号由幅度项承担，必须联读。',
            '- parseval的完整总误差等价于空间MSE，其价值是可核对的频段/幅度—相位归因，不是凭空新增总体信息。',
            '- 无窗按有限数组的周期延拓解释频率，边界不连续会影响频段归因；其总MSE恒等式不受影响。',
            '- 基础版Hann口径与parseval无窗口径分别报告，不能直接将两者频段能量相减比较。',
            '- 单位为索引频率cycles/pixel。没有物理波数、泛化、显著性或独立盲测结论。',
            '- 每项增强均可单独关闭；没有综合总分或根据本批结果挑选指标。',
            '', '所有逐帧、分组和汇总数据见metrics.json；每个增强另存自己的metrics.json和spectra.npz。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

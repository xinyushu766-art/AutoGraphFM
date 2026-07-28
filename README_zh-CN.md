# AutoGraphFM

[English](README.md)

AutoGraphFM 是一个面向汽车销量预测的可审计机器学习系统，覆盖：

```text
中英文字段自适应清洗
→ 数据质量审计与固定节点月度面板
→ 统计 / 机器学习 / 深度时序模型
→ 静态图、自适应图与稳定稀疏图
→ 泄漏安全滚动回测
→ 模型选优与区间预测
→ 数据漂移和发布门禁
→ 真实结果回流与回滚决策
```

## 项目亮点

- 建立从真实业务 CSV 到预测发布的完整 ML 生命周期，而非单一训练脚本。
- 实现 Last Value、Seasonal Naive、Ridge、NLinear/DLinear 风格模型、GRU、Regime MoE、AGCRN/MTGNN 风格图和稳定稀疏图。
- 建立 **108 项自动化测试**，覆盖数据、数据库、回测、图学习、监控、发布治理和报告。
- 在 2018—2024 中国汽车市场公开聚合数据上，新模型在 **6 个目标×周期任务中的4个取得改善**；新能源3个月预测 WAPE 相对下降 **9.81%**。
- 在受控已知图实验中，稳定图候选边 Precision@K 达到 **83.3%**，跨随机种子边 Jaccard 达到 **1.00**。
- 设置时间模型、静态图、单位图、置乱图和配对 Bootstrap 对照；没有真实车型证据时保持 `NO-CLAIM`，避免夸大结论。

## 快速运行

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_stage10b.py --output-dir outputs/stage10b_real --overwrite
python scripts/validate_stage12a.py --overwrite --full
```

## 关键结果

| 任务 | 结果 |
|---|---:|
| 新能源1个月预测 WAPE | 18.82% |
| 新能源3个月预测 WAPE | 24.80% |
| 全市场3个月预测 WAPE | 15.63% |
| 全市场6个月预测 WAPE | 16.43% |
| 受控图恢复 Precision@K | 83.3% |
| 最终边跨种子 Jaccard | 1.00 |
| 自动化测试 | 108/108 |

完整方法、证据边界和面试讲解见 [`docs/`](docs/)。

## 作者

**Xinyu Shu（舒新宇）** — [@xinyushu766-art](https://github.com/xinyushu766-art)

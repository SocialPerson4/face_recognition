# Face Verification under High Identity Similarity

面向高相似身份风险的人脸验证研究。项目从传统机器学习出发，研究 PCA 特征维数、距离度量和支持向量机对人脸验证性能与错误接受风险的影响。

## Research scope

- 第一阶段：在 ORL 数据集上建立无泄漏的人脸验证基线并选择参数。
- 第二阶段：比较普通负样本与高相似困难负样本的错误接受风险。
- 第三阶段：在 LFW Funneled 上检查方法在复杂图像条件下的表现。
- 双胞胎是研究动机和未来验证场景，当前不声称已完成双胞胎实验。

## Current status

- M0 research protocol：已完成
- M1 ORL data audit and pairing protocol：已完成
- M2A verification metrics：已完成
- M2B ORL distance baseline：已完成单次开发实验，最终测试集仍封存
- M3A PCA 维数粗搜索：已完成，候选平台区为 80—180 维
- M3B PCA 维数细搜索：已完成，距离基线按一标准误差规则选择 80 维
- M3C 固定维数模型家族初筛：已完成，默认学习模型出现明显过拟合
- M3D 分类器超参数粗搜索：已完成，强正则化明显改善但最优C位于下边界
- M3E 分类器下边界扩展：已完成，强正则化区出现稳定排序平台
- M3F 更低C哨兵确认：已完成，未发现稳定实质改善，候选参数已冻结
- M4 冻结候选开发验证：已完成，RBF按EER-AUC规则成为开发候选，距离在低FMR工作点更优
- M5 一次性最终测试：待开展
- Python 为唯一正式实现
- MATLAB 和原课程报告不进入本仓库
- 原始人脸图片不会提交 GitHub

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/audit_orl.py
.venv/bin/python scripts/build_orl_protocol.py
.venv/bin/python scripts/check_metrics.py
.venv/bin/python scripts/run_orl_distance_baselines.py
.venv/bin/python scripts/search_orl_pca_k.py
.venv/bin/python scripts/search_orl_pca_k.py --stage fine --k-values 80 100 120 140 160 170 180 185 --output-dir results/experiments/orl_pca_k_fine_seed_20260913
.venv/bin/python scripts/plot_pca_k_search.py
.venv/bin/python scripts/plot_pca_k_fine_selection.py
.venv/bin/python scripts/compare_orl_model_families.py
.venv/bin/python scripts/plot_orl_model_screening.py
.venv/bin/python scripts/search_orl_classifier_hyperparameters.py
.venv/bin/python scripts/plot_orl_classifier_search.py
.venv/bin/python scripts/search_orl_classifier_hyperparameters.py --stage boundary
.venv/bin/python scripts/check_classifier_search_overlap.py
.venv/bin/python scripts/plot_orl_classifier_search.py --input results/experiments/orl_classifier_boundary_k80_seed_20260913/summary.json --output-dir results/figures/orl_classifier_boundary_k80_seed_20260913
.venv/bin/python scripts/search_orl_classifier_hyperparameters.py --stage sentinel
.venv/bin/python scripts/check_classifier_search_overlap.py --reference results/experiments/orl_classifier_boundary_k80_seed_20260913/summary.json --candidate results/experiments/orl_classifier_sentinel_k80_seed_20260913/summary.json --output results/experiments/orl_classifier_sentinel_k80_seed_20260913/overlap_audit.json
.venv/bin/python scripts/plot_orl_classifier_search.py --input results/experiments/orl_classifier_sentinel_k80_seed_20260913/summary.json --output-dir results/figures/orl_classifier_sentinel_k80_seed_20260913
.venv/bin/python scripts/evaluate_orl_frozen_candidates.py
.venv/bin/python scripts/plot_orl_frozen_validation.py
.venv/bin/python -m pytest -q
```

ORL 数据应放在 `data/raw/orl/`，具体结构见 [data/README.md](data/README.md)。

## Documentation

- [Research plan](docs/research_plan.md)
- [Research thinking and decisions](docs/research_thinking.md)
- [Code and module guide](docs/code_guide.md)

复试演讲词与个人准备统一保存在本机 `local/interview_prep.md`，该目录不会提交 GitHub。

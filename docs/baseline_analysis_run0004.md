# Perch Fixed Baseline Analysis (run-0004)

## Experiment Results

**Primary Metric**: `val_soundscape_macro_roc_auc = 0.6649`

### All Metrics
- Val holdout AUC: **0.6649** (primary)
- Val prior fusion: 0.6622
- Resubstitution: 0.9918
- Prior fusion OOF: 0.4873
- OOF probe: 0.5201
- Padded cMAP: 0.0623

### Dataset
- 708 windows, 59 files
- 75 active classes
- 52 fitted classes (train)
- 34 fitted classes (holdout)

### Holdout Split
- Train: 47 files (564 windows)
- Val: 12 files (144 windows)
- Sites: S03, S08, S13, S19

## Key Findings

### 1. Probe增益极小
```
Val prior fusion: 0.6622
Val full pipeline: 0.6649
Probe gain: 0.0027 (0.4%)
```

### 2. 与参考实验对比
| Exp | Primary Metric | Value | Method |
|-----|----------------|-------|--------|
| harness_ref SED-v9 | Holdout AUC | 0.9467 | EfficientNet-B0 + ASL |
| harness_ref ensemble | Kaggle LB | 0.893 | Perch×3 + SED + tricks |
| Current Perch | Val holdout | 0.6649 | Perch probe only |

**Gap**: 0.9467 - 0.6649 = **0.28** (SED比Perch probe高28个点)

### 3. 根本原因
- Perch probe仅用708个soundscape窗口训练
- SED用35,549个train_audio训练
- 训练数据量差异: **50倍**

## Recommendations

### Option A: 超参调优 (快速)
基于harness_ref知识库的最佳实践:

```yaml
# 当前配置
probe_pca_dim: 32  → 尝试: 64, 128
probe_min_pos: 8   → 尝试: 4, 6
probe_c: 0.25      → 尝试: 0.1, 0.5, 1.0
probe_alpha: 0.40  → 尝试: 0.3, 0.5, 0.6
```

预期增益: +0.01~0.03

### Option B: 生成Kaggle提交 (推荐)
- Verdict: `submission-required`
- 当前val AUC (0.6649) 不代表Kaggle LB分数
- Notebook的0.912是LB分数，本地无法复现
- 需要提交到Kaggle获取真实性能

### Option C: 集成SED模型
参考harness_ref的ensemble策略:
- Perch probe (当前) + SED-B0 (需训练)
- 预期LB: 0.89+ (基于harness_ref经验)

## Next Steps

1. **立即**: 生成submission并提交到Kaggle
2. **并行**: 启动超参网格搜索
3. **长期**: 训练SED模型用于ensemble


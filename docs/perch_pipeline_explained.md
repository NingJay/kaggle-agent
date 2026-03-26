# Perch Pipeline 运行流程与评估指标详解

## 1. 整体架构

```
Perch Embeddings (预计算)
    ↓
Prior Fusion (贝叶斯先验融合)
    ↓
Probe Models (逻辑回归探针)
    ↓
Gaussian Smoothing (高斯平滑)
    ↓
Final Predictions
```

## 2. 数据流程

### 2.1 输入数据
- **Perch embeddings**: 1536维向量，从预训练的Perch模型提取
- **Raw scores**: Perch模型的原始分类分数 (182类)
- **Soundscape labels**: 708个完全标注的5秒窗口
  - 59个文件 × 12个窗口/文件 = 708个窗口
  - 75个活跃类别（在soundscape中出现过）

### 2.2 训练流程

#### Step 1: Prior Fusion (先验融合)
```python
# 构建先验表
site_hour_tables = build_prior_tables(soundscape_data)
# 包含:
# - site_event_prior: 站点×事件类别的先验概率
# - site_texture_prior: 站点×纹理类别的先验概率
# - hour_event_prior: 小时×事件类别的先验概率
```

**融合公式**:
```
fused_score = (1-λ) × raw_score + λ × prior_score
```
- Event类 (鸟类): λ_event = 0.4
- Texture类 (昆虫/两栖): λ_texture = 1.0

#### Step 2: Probe Training (探针训练)
```python
# 对每个类别:
# 1. PCA降维: 1536维 → 32维
# 2. 构建特征:
#    - PCA embeddings
#    - Prior fusion scores
#    - Sequential features (窗口内上下文)
#    - Prototype similarity (类原型相似度)
#    - Family mean (科级平均特征)
# 3. 训练逻辑回归: LogisticRegression(C=0.25, class_weight='balanced')
```

**训练条件**:
- 最少正样本数: min_pos = 8
- 结果: 52/75 类别成功训练探针

#### Step 3: Prediction & Smoothing
```python
# 1. Probe预测
probe_pred = model.predict_proba(features)

# 2. 与prior fusion混合
final = (1-α) × prior_fusion + α × probe_pred  # α=0.4

# 3. 高斯平滑 (按文件内12个窗口)
smoothed = gaussian_filter(final, weights=[0.1,0.2,0.4,0.2,0.1])
```

## 3. 评估指标详解

### 3.1 Primary Metric: `val_soundscape_macro_roc_auc`
**值**: 0.6729

**含义**: Holdout验证集上的macro-averaged ROC-AUC

**计算方式**:
```python
# 1. 站点级拆分 (避免数据泄漏)
train_sites = [S01, S02, S04, ...] # 47 files, 564 windows
val_sites = [S03, S08, S13, S19]   # 12 files, 144 windows

# 2. 仅用train数据训练完整pipeline
prior_tables = fit_prior(train_soundscapes)
probe_models = fit_probes(train_embeddings, train_labels)

# 3. 对val数据预测
val_preds = full_pipeline.predict(val_embeddings)

# 4. 计算macro AUC (仅对有正样本的类)
aucs = [roc_auc(val_labels[:, i], val_preds[:, i])
        for i in active_classes if val_labels[:, i].sum() > 0]
val_auc = mean(aucs)  # 0.6729
```

**用途**:
- ✅ 本地超参调优的主要指标
- ✅ 不同配置的相对排序
- ❌ 不能与SED实验直接比较 (训练数据量差100倍)

---

### 3.2 Secondary Metrics

#### `soundscape_macro_roc_auc` = 0.9918
**含义**: Resubstitution分数 (训练集自评)

**计算方式**:
```python
# 全量训练 (708个窗口)
models = fit_full_pipeline(all_data)
# 在训练集上评估
train_preds = models.predict(all_data)
auc = macro_roc_auc(all_labels, train_preds)  # 0.9918
```

**用途**:
- ✅ 验证代码正确性 (应该接近1.0)
- ❌ 不能用于模型改进 (过拟合指标)

---

#### `prior_fusion_macro_roc_auc` = 0.4845
**含义**: Prior fusion baseline (不含probe)

**计算方式**:
```python
# OOF评估 (5-fold cross-validation)
for fold in folds:
    prior_tables = fit_prior(train_folds)
    oof_preds[val_fold] = apply_prior_fusion(val_fold, prior_tables)
auc = macro_roc_auc(all_labels, oof_preds)  # 0.4845
```

**用途**:
- ✅ Baseline参考 (probe应该比这个高)
- ✅ 验证prior fusion是否工作

---

#### `oof_probe_macro_roc_auc` = 0.5192
**含义**: Probe的OOF分数 (5-fold CV)

**计算方式**:
```python
# 对每个fold:
for fold in folds:
    probe_models = fit_probes(train_folds)
    oof_preds[val_fold] = probe_models.predict(val_fold)
auc = macro_roc_auc(all_labels, oof_preds)  # 0.5192
```

**特点**:
- 比prior fusion高 (0.5192 > 0.4845) ✓
- 但比holdout validation低 (0.5192 < 0.6729)
- 原因: OOF更保守，每个fold只用80%数据训练

---

#### `val_prior_fusion_macro_roc_auc` = 0.6699
**含义**: Holdout验证集上的prior fusion分数

**用途**: 对比probe的增益
```
probe增益 = 0.6729 - 0.6699 = 0.003 (0.4%)
```
说明probe在holdout上几乎没有提升，可能需要调整超参。

---

#### `padded_cmap` = 0.0625
**含义**: Padded class-averaged mean average precision

**计算方式**:
```python
# 对每个类别计算AP，未出现的类用0填充
aps = [average_precision(y_true[:, i], y_pred[:, i])
       for i in all_182_classes]
cmap = mean(aps)  # 包含很多0值
```

**特点**:
- 非常低 (0.0625) 因为182个类中只有75个活跃
- 107个类的AP=0拉低了平均值
- 不是主要优化目标

## 4. 指标对比表

| 指标 | 值 | 训练方式 | 评估方式 | 用途 |
|------|-----|----------|----------|------|
| `val_soundscape_macro_roc_auc` | **0.6729** | Train sites | Val sites | **主指标** - 超参调优 |
| `soundscape_macro_roc_auc` | 0.9918 | 全量 | 训练集 | 代码验证 |
| `prior_fusion_macro_roc_auc` | 0.4845 | OOF | OOF | Baseline参考 |
| `oof_probe_macro_roc_auc` | 0.5192 | OOF | OOF | Probe OOF性能 |
| `val_prior_fusion_macro_roc_auc` | 0.6699 | Train sites | Val sites | 对比probe增益 |
| `padded_cmap` | 0.0625 | 全量 | 训练集 | 次要指标 |

## 5. 与Notebook的对应关系

### Notebook的0.912是什么？
**Kaggle Public LB分数** (提交到Kaggle后的排行榜分数)

**Notebook流程**:
```python
# 1. Prior fusion OOF
oof_auc = 0.487  # Notebook打印的唯一本地AUC

# 2. 全量训练probe (无OOF)
probe_models = fit_probes(all_708_windows)

# 3. 对test_soundscapes推理
test_preds = full_pipeline.predict(test_embeddings)

# 4. 提交到Kaggle
submit(test_preds)  # → Public LB = 0.912
```

**为什么本地无法复现0.912？**
- Notebook从不计算probe的本地AUC
- 0.912是test set上的分数，本地没有test labels
- 唯一验证方式：生成submission提交到Kaggle

### 当前系统的对应
```
Notebook prior fusion OOF (0.487)
  ≈ 当前 prior_fusion_macro_roc_auc (0.4845) ✓

Notebook Kaggle LB (0.912)
  ≈ 当前 ??? (需要提交到Kaggle才知道)

当前 val_soundscape_macro_roc_auc (0.6729)
  = 新增的holdout validation (Notebook没有)
```

## 6. 改进方向

### 6.1 基于 `val_soundscape_macro_roc_auc` 调优
```yaml
# 当前配置
probe_pca_dim: 32        # 可尝试: 64, 128
probe_min_pos: 8         # 可尝试: 4, 6
probe_c: 0.25            # 可尝试: 0.1, 0.5, 1.0
probe_alpha: 0.40        # 可尝试: 0.3, 0.5, 0.6
```

### 6.2 分析probe增益低的原因
```
val_prior_fusion: 0.6699
val_full_pipeline: 0.6729
增益: 0.003 (0.4%)
```

可能原因:
1. Holdout数据太少 (144 windows)
2. 52个fitted classes中，holdout只有34个有正样本
3. Probe过拟合训练集

### 6.3 最终验证
```bash
# 生成submission
python inference.py --config configs/default.yaml

# 提交到Kaggle
kaggle competitions submit -f submission.csv

# 查看LB分数 (期望 ~0.91)
```

## 7. 总结

**关键理解**:
1. `val_soundscape_macro_roc_auc = 0.6729` 是当前**唯一可用于本地迭代的指标**
2. Notebook的0.912是Kaggle LB分数，本地无法复现
3. Perch pipeline的真实性能只能通过Kaggle提交验证
4. 当前holdout validation用于快速超参调优，不代表最终LB分数

**Verdict**: `"submission-required"`
→ 需要生成submission并提交到Kaggle获取真实分数

# P11_Regression_Mix.ipynb 总体内容总结及后续修改建议

## 📋 项目概述

本项目是一个基于**混合回归模型**的S&P 500指数OHLC（开高低收）价格预测系统。系统采用模块化特征工程、多种机器学习模型和混合预测策略，通过严格的实验框架评估不同特征集和模型组合的性能。

### 核心目标
- 预测S&P 500指数未来30天的OHLC价格
- 评估不同特征集和模型组合的预测性能
- 提供不确定性量化（预测区间）
- 支持混合预测策略（迭代+多步预测）

---

## 🏗️ 代码结构

### 1. 数据获取模块（Cell 1-5）

#### Cell 1: 导入库
- 核心库：pandas, numpy, yfinance, sklearn
- 模型：GradientBoosting, RandomForest, Ridge, Lasso, ElasticNet, KNN, XGBoost
- 工具：特征工程、评估指标、可视化

#### Cell 2: S&P 500数据获取
- 数据源：`^GSPC` (2010-01-01 至今)
- 列名标准化：Open→O, High→H, Low→L, Close→C
- **已修复**：添加 `auto_adjust=True` 消除警告

#### Cell 3: 债券数据获取
- 数据源：`^TYX` (30年期美债收益率)
- 数据对齐：与S&P 500数据按日期合并
- 前向填充：处理缺失日期
- **已修复**：添加 `auto_adjust=True` 消除警告

#### Cell 4: 宏观指标获取
- USDJPY：美元/日元汇率
- XAUUSD：黄金期货价格
- HSI：恒生指数
- VIX：CBOE波动率指数
- **已修复**：添加 `auto_adjust=True` 消除警告

#### Cell 5: 期货数据获取
- ES：S&P 500 E-mini期货
- NQ：Nasdaq E-mini期货
- ZB：30年期美债期货
- VIX_Futures：VIX期货（可能失败）
- **已修复**：添加 `auto_adjust=True` 消除警告

---

### 2. 预测方法模块（Cell 6, 8）

#### Cell 6: 混合预测方法 (`forecast_future_prices_hybrid`)
**核心特性**：
- **阶段1**：前N天（默认7天）迭代预测，保证短期精度
- **阶段2**：后M天（默认23天）多步预测，减少累积误差
- **不确定性量化**：支持3种方法
  - `quantile_regression`：分位数回归（推荐）
  - `residual_bootstrap`：残差Bootstrap
  - `ensemble_variance`：集成方法方差
- **期货数据整合**：使用ES期货等作为外部变量预测

**已修复**：
- ✅ 添加 `futures_data` 和 `macro_data_dict` 参数，移除全局变量依赖
- ✅ 统一外部变量列表：`["Bond30Y", "USDJPY", "XAUUSD", "HSI", "VIX"]`

#### Cell 8: 基础迭代预测 (`forecast_future_prices`)
**核心特性**：
- 逐日迭代预测未来30天
- 每次预测后更新数据框，用于下一日特征构建
- OHLC约束校准（L≤O/H/C，H≥O/L/C）

**已修复**：
- ✅ 移除废弃的 `fillna(method=...)` 代码
- ✅ 统一外部变量列表（添加VIX）

---

### 3. 特征工程模块（Cell 7）

#### 基础特征
- `add_calendar_features`：年、月、日、星期
- `add_bond_feature`：30年期美债收益率（前向填充）

#### 时序特征
- `add_lag_features`：滞后特征（1, 5, 10天）
- `add_return_features`：收益率特征（对数/百分比）
- `add_moving_average_features`：移动平均（5, 20, 60天）
- `add_rsi_feature`：RSI相对强弱指标
- `add_volatility_features`：波动率特征

#### 量价关系特征
- `add_volume_price_features`：
  - 相对成交量（Volume Ratio）
  - 量价相关性
  - OBV（On-Balance Volume）
  - VWAP（成交量加权平均价格）
  - 异常成交量检测

#### VIX波动率特征
- `add_vix_features`：
  - VIX水平值、变化率
  - VIX移动平均、分位数排名
  - VIX与价格的相关性
  - 高/低波动信号

#### 宏观特征
- `add_macro_close`：USDJPY, XAUUSD, HSI, VIX（滞后1期）

#### 长期趋势特征
- `add_long_term_moving_average`：长周期SMA（120, 250, 500天）
- `add_long_lag_features`：长周期滞后（60, 120, 250天）
- `add_ma_crossover_features`：短期/长期均线交叉

#### 非线性特征
- `add_pairwise_interactions`：两两特征交互项
- `add_polynomial_features`：多项式特征（3次）
- `add_spline_features`：样条函数特征（3次，5-6个节点）

#### 特征构建核心函数
- `build_feature_frame`：按顺序应用特征构建函数，返回特征DataFrame

---

### 4. 模型系统（Cell 11）

#### 支持的模型
1. **GradientBoosting**：梯度提升回归
2. **RandomForest**：随机森林（300棵树）
3. **KNN**：K近邻（带标准化）
4. **Ridge**：岭回归（带标准化）
5. **Lasso**：Lasso回归（带标准化）
6. **ElasticNet**：弹性网络（带标准化）
7. **XGBoost**：XGBoost回归（500棵树，学习率0.05）

所有模型使用 `MultiOutputRegressor` 包装，支持同时预测OHLC四个目标。

---

### 5. 实验评估框架（Cell 12-13）

#### Cell 12: 严格时间序列划分
- `strict_time_series_split`：
  - 确保训练集 < 验证集 < 测试集（按时间顺序）
  - 避免数据泄漏
  - 支持二划分或三划分

- `build_features_with_lookback`：
  - 训练集特征：仅使用训练集数据计算
  - 测试集特征：使用训练集末尾+测试集数据计算
  - 滚动窗口特征：在测试集计算时，使用训练集末尾数据作为lookback

#### Cell 13: 实验执行函数
- `run_experiment`：
  - 支持严格时间序列划分（推荐）
  - 自动特征标准化
  - 多模型评估（MAE, RMSE, R²）
  - 返回预测结果和评估指标

---

### 6. 特征集配置（Cell 14）

定义了**13个特征集配置**，从基础到复杂：

1. **calendar_bond**：日历特征 + 债券
2. **calendar_bond_lags**：+ 滞后特征
3. **calendar_lags_returns**：+ 收益率
4. **technical_momentum**：+ SMA, RSI, 波动率
5. **technical_macro**：+ 宏观指标（USDJPY, XAUUSD, HSI）
6. **volume_price_features**：+ 量价关系特征
7. **vix_features**：+ VIX波动率特征
8. **enhanced_macro_volume**：完整特征集（宏观+VIX+量价）
9. **macro_only**：仅宏观指标
10. **pairwise_interactions**：+ 交互项
11. **hierarchical_polynomial**：+ 多项式特征
12. **cubic_splines**：+ 三次样条
13. **natural_splines**：+ 自然样条

---

### 7. 实验执行与结果分析（Cell 15-28）

#### Cell 15: 批量实验执行
- 对每个特征集 × 模型组合进行实验
- 收集所有实验结果

#### Cell 16-17: 结果汇总与可视化
- 按特征集和模型汇总MAE、RMSE、R²
- 生成性能对比图表

#### Cell 18-22: 预测可视化
- 绘制历史价格与预测对比
- 多目标（OHLC）预测可视化
- 模型性能对比图

#### Cell 23-28: 最佳组合选择
- 选择RMSE最低的特征集+模型组合
- 实际vs预测对比图
- 相关性热力图

---

## ✅ 已完成的修复

### 高优先级修复

#### 1. 修复 `fillna(method=...)` 废弃方法
- **位置**：Cell 8 (`forecast_future_prices`)
- **修复**：删除废弃的 `fillna(method="ffill")` 和 `fillna(method="bfill")`
- **改为**：直接使用 `ffill()` 和 `bfill()` 方法

#### 2. 修复 yfinance 警告
- **位置**：Cell 2, 3, 4, 5
- **修复**：在所有 `yf.download()` 调用中显式添加 `auto_adjust=True`
- **效果**：消除 FutureWarning 警告

### 中优先级修复

#### 3. 统一外部变量列表
- **位置**：Cell 8 (`forecast_future_prices`)
- **修复**：添加 `"VIX"` 到外部变量列表
- **统一为**：`["Bond30Y", "USDJPY", "XAUUSD", "HSI", "VIX"]`

#### 4. 改进参数传递方式
- **位置**：Cell 6 (`forecast_future_prices_hybrid`)
- **修复**：
  - 添加 `futures_data` 和 `macro_data_dict` 参数
  - 移除对 `globals()` 的依赖
  - 所有数据通过函数参数传递
  - 修复函数内部所有 `macro_data` 引用

---

## 🔧 后续修改建议

### 高优先级（必须修复）

#### 1. 修复 `build_feature_frame` 中的函数名检查问题
**问题**：
```python
if func.__name__ == 'add_macro_close':
```
当使用 `functools.partial` 时，`__name__` 属性不可用，会导致 `AttributeError`。

**建议修复**：
```python
# 方法1：使用函数对象本身判断
if func.func.__name__ == 'add_macro_close' if isinstance(func, partial) else func.__name__ == 'add_macro_close':

# 方法2：更优雅的方式 - 检查函数签名
import inspect
sig = inspect.signature(func.func if isinstance(func, partial) else func)
if 'macro_data_dict' in sig.parameters:
    df_features, new_cols = func(df_features, macro_data_dict=kwargs.get('macro_data_dict', {}), **kwargs)
else:
    df_features, new_cols = func(df_features, **kwargs)
```

**位置**：Cell 7 (`build_feature_frame` 函数)

---

#### 2. 修复 `forecast_future_prices` 中缺少 `macro_data_dict` 参数
**问题**：
`forecast_future_prices` 函数调用 `build_feature_frame` 时未传递 `macro_data_dict`，导致宏观特征无法正确构建。

**建议修复**：
```python
def forecast_future_prices(
    base_df: pd.DataFrame,
    feature_funcs: List[FeatureBuilder],
    model_factory: Callable[[], MultiOutputRegressor],
    horizon: int = 30,
    macro_data_dict: Dict[str, pd.DataFrame] = None,  # 新增参数
) -> Tuple[pd.DataFrame, MultiOutputRegressor, List[str]]:
    # ...
    train_features, feature_columns = build_feature_frame(
        working_df, feature_funcs, macro_data_dict=macro_data_dict or {}
    )
    # ...
    feature_frame_full, _ = build_feature_frame(
        working_df, feature_funcs, dropna=False, macro_data_dict=macro_data_dict or {}
    )
```

**位置**：Cell 8

---

### 中优先级（建议修复）

#### 3. 统一预测函数接口
**问题**：
- `forecast_future_prices` 和 `forecast_future_prices_hybrid` 参数不一致
- `forecast_future_prices` 缺少 `macro_data_dict` 参数

**建议**：
- 为 `forecast_future_prices` 添加 `macro_data_dict` 参数
- 统一两个函数的参数命名和顺序
- 考虑将 `forecast_future_prices` 作为 `forecast_future_prices_hybrid` 的简化版本

---

#### 4. 改进错误处理
**问题**：
- 数据获取失败时缺少重试机制
- 特征构建失败时错误信息不够详细

**建议**：
- 为数据获取添加重试机制（最多3次）
- 增加详细的错误日志
- 特征构建失败时输出具体是哪个特征函数失败

---

#### 5. 优化性能
**问题**：
- 每次迭代都调用 `build_feature_frame`，开销较大
- 多步预测模型每次预测都重新训练

**建议**：
- 考虑批量构建特征或缓存中间结果
- 多步预测模型可以提前训练并复用
- 使用 `numba` 或 `cython` 加速特征计算

---

### 低优先级（可选优化）

#### 6. 提取配置常量
**问题**：
- 硬编码的 magic numbers：`iterative_days=7`, `horizon=30`, `lookback_window=252`

**建议**：
```python
# 在文件开头定义配置常量
PREDICTION_CONFIG = {
    "iterative_days": 7,
    "horizon": 30,
    "lookback_window": 252,
    "test_size": 0.2,
}
```

---

#### 7. 完善文档字符串
**问题**：
- 部分函数缺少完整的 docstring
- 复杂逻辑缺少注释

**建议**：
- 为所有公共函数添加完整的 docstring（参数、返回值、示例）
- 为复杂算法添加注释说明

---

#### 8. 增强类型提示
**问题**：
- 部分函数返回值类型提示缺失或不准确

**建议**：
- 使用 `typing` 模块完善类型提示
- 考虑使用 `typing_extensions` 的新特性

---

#### 9. 功能完善

##### 9.1 混合预测方法集成到实验框架
**问题**：
- `forecast_future_prices_hybrid` 已定义但未在 `run_experiment` 中使用
- Cell 18-20 仍使用旧的 `forecast_future_prices`

**建议**：
- 在 `run_experiment` 中添加选项支持混合预测
- 更新预测可视化代码使用混合预测方法

##### 9.2 不确定性量化可视化
**问题**：
- `forecast_future_prices_hybrid` 返回不确定性区间，但未绘制

**建议**：
- 添加不确定性区间可视化（置信区间带）
- 显示预测的不确定性随时间的变化

##### 9.3 模型持久化
**问题**：
- 训练好的模型未保存，每次都需要重新训练

**建议**：
- 添加模型保存/加载功能（使用 `pickle` 或 `joblib`）
- 保存最佳模型配置和特征列表

---

#### 10. 代码质量改进

##### 10.1 减少代码重复
**问题**：
- `forecast_future_prices` 和 `forecast_future_prices_hybrid` 有重复代码

**建议**：
- 提取公共逻辑到辅助函数
- 使用组合模式减少重复

##### 10.2 改进测试
**问题**：
- 缺少单元测试

**建议**：
- 为关键函数添加单元测试
- 使用 `pytest` 框架

---

## 📊 关键参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据起始日期 | 2010-01-01 | S&P 500历史数据起始点 |
| 预测时间范围 | 30天 | 未来预测天数 |
| 迭代预测天数 | 7天 | 混合预测中迭代预测的天数 |
| 测试集比例 | 20% | 时间序列划分比例 |
| Lookback窗口 | 252天 | 滚动窗口特征的历史数据窗口 |
| 目标变量 | ["H", "L", "O", "C"] | OHLC四个价格目标 |
| 外部变量 | ["Bond30Y", "USDJPY", "XAUUSD", "HSI", "VIX"] | 外部宏观经济变量 |

---

## 🎯 核心设计特点

### 1. 模块化特征工程
- 每个特征类型独立函数
- 可组合的特征构建器
- 易于扩展新特征

### 2. 严格的数据泄漏防护
- 严格时间序列划分
- 特征构建时使用lookback窗口
- 外部变量强制滞后处理

### 3. 多目标预测
- 同时预测OHLC四个目标
- 使用 `MultiOutputRegressor` 包装
- OHLC约束校准

### 4. 混合预测策略
- 短期迭代预测（保证精度）
- 长期多步预测（减少误差累积）
- 不确定性量化

### 5. 全面的实验框架
- 13个特征集配置
- 7种模型选择
- 自动性能评估和可视化

---

## 📈 工作流程

```
数据获取 → 特征工程 → 模型训练 → 预测 → 评估 → 可视化
   ↓          ↓          ↓         ↓      ↓       ↓
Cell 2-5   Cell 7    Cell 11   Cell 6,8  Cell 13  Cell 16-28
```

---

## 🔍 已知问题

1. **Cell 15 执行时可能报错**：`build_feature_frame` 中的函数名检查问题（见修复建议1）
2. **宏观特征可能无法正确构建**：`forecast_future_prices` 缺少 `macro_data_dict` 参数（见修复建议2）
3. **混合预测方法未集成**：Cell 18-20 仍使用旧方法（见修复建议9.1）

---

## 📝 总结

本项目是一个功能完整的股票价格预测系统，具有以下优势：
- ✅ 模块化设计，易于扩展
- ✅ 严格的数据泄漏防护
- ✅ 多种预测策略和不确定性量化
- ✅ 全面的实验评估框架

**已完成的修复**：
- ✅ 修复废弃API警告
- ✅ 统一外部变量列表
- ✅ 改进参数传递方式

**待修复的关键问题**：
- ⚠️ `build_feature_frame` 函数名检查问题
- ⚠️ `forecast_future_prices` 缺少 `macro_data_dict` 参数
- ⚠️ 混合预测方法未集成到实验框架

建议优先修复高优先级问题，然后逐步完善中低优先级功能。


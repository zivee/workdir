# combined_train_cpu_inference_focused.py
# ==============================================================================
# ## Mihomo 智能权重模型训练 (CPU版 - 聚焦推理性能)
# ---
#
# ### **核心策略**
# 本脚本专注于算法层面的优化，旨在训练出一个既能保持高预测精度，
# 又拥有极快推理速度的最终模型。它移除了所有GPU相关逻辑，确保在纯CPU环境下的
# 普适性。
# 1.  **探索阶段 (寻找性能拐点)**: 使用 Optuna 和 K-Fold 交叉验证。目标函数
#     经过精心设计，引入一个微小的“复杂度惩罚”作为 tie-breaker (决胜局标准)。
#     当多个超参数组合都能达到相似的高精度时，该机制会优先选择那个能产生
#     最简单（因此最快）模型的组合。
# 2.  **训练阶段**: 利用找到的最佳超参数，在 **全部数据集** 上训练一个最终的、
#     单一的模型。
# ==============================================================================
import re
import os
import shutil
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import Tuple, List, Optional

# ==============================================================================
# 1. 全局配置 (Global Configuration)
# ==============================================================================
DATA_FILE = 'smart_weight_data.csv'
GO_FILE = 'transform.go'
MODEL_DIR = 'models'
FINAL_MODEL_NAME = 'Model.bin'

# --- 训练与优化控制 ---
OPTUNA_TRIALS = 50
KFOLD_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50

# --- 特征工程与预处理 ---
OUTLIER_CLIP_QUANTILE = 0.99
STD_SCALER_FEATURES = ['connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 'last_used_seconds', 'traffic_density']
ROBUST_SCALER_FEATURES = ['success', 'failure']

# --- 【优化】超参数搜索空间 (为高精度和高效率平衡设计) ---
# 相比纯速度优化，我们适当放宽了 num_leaves 和 max_depth 的上限，
# 确保优化器有足够空间找到高精度模型。
PARAM_GRID = {
    "n_estimators": (300, 2000),
    "learning_rate": (0.01, 0.1),
    "num_leaves": (20, 80),      # 适当放宽上限，确保能达到高精度
    "max_depth": (5, 15),       # 适当放宽上限
    "min_child_samples": (20, 100),
    "reg_alpha": (1e-3, 10.0),  # L1 正则化
    "reg_lambda": (1e-3, 10.0)  # L2 正则化
}

# --- 【优化】目标函数权重 ---
# 核心：COMPLEXITY_WEIGHT 设置得非常小，仅作为高精度模型间的“决胜局”裁判。
# 它不会强迫模型牺牲精度来换取速度，而是在精度相近时，选择更快的那个。
R2_WEIGHT = 1.0
RMSE_WEIGHT = 0.1
COMPLEXITY_WEIGHT = 0.0005 # 【关键调整】权重很小，扮演 tie-breaker 角色

# ==============================================================================
# 2. 核心功能函数 (Core Functions)
# ==============================================================================
# (GoTransformParser, load_and_clean_data, clip_weight_outliers, create_feature_pipeline 等函数保持不变，此处省略)
class GoTransformParser:
    def __init__(self, go_file_path: str):
        try:
            with open(go_file_path, 'r', encoding='utf-8') as f: self.content = f.read()
            print(f"成功加载 Go 源文件: {go_file_path}")
        except FileNotFoundError: raise FileNotFoundError(f"Go 源文件 '{go_file_path}' 没找到。")
    def get_feature_order(self) -> List[str]:
        print("开始解析 getDefaultFeatureOrder 函数...")
        p = r'func getDefaultFeatureOrder\(\) map\[int\]string \{\s*return map\[int\]string\{(.*?)\}\s*\}'
        m = re.search(p, self.content, re.DOTALL)
        if not m: return self._get_fallback_feature_order()
        b = m.group(1)
        pairs = re.findall(r'(\d+):\s*"([^"]+)"', b)
        if not pairs: return self._get_fallback_feature_order()
        d = {int(i): n for i, n in pairs}
        return [d[i] for i in sorted(d.keys())]
    def _get_fallback_feature_order(self) -> List[str]: return ['success', 'failure', 'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 'last_used_seconds', 'is_udp', 'is_tcp', 'asn_feature', 'country_feature', 'address_feature', 'port_feature', 'traffic_ratio', 'traffic_density', 'connection_type_feature', 'asn_hash', 'host_hash', 'ip_hash', 'geoip_hash']

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    print(f"开始加载数据文件: {file_path}")
    try: data = pd.read_csv(file_path, on_bad_lines='skip')
    except FileNotFoundError: print(f"错误: 数据文件 '{file_path}' 不存在"); return None
    print(f"数据加载完成，原始记录数: {len(data)}")
    original_count = len(data)
    data.dropna(subset=['weight'], inplace=True)
    data = data[data['weight'] > 0].copy()
    print(f"数据清洗完成: {original_count} → {len(data)} 条记录 (过滤 {original_count - len(data)} 条)")
    return data

def clip_weight_outliers(y: pd.Series, quantile: float) -> Tuple[pd.Series, float]:
    clip_threshold = y.quantile(quantile)
    y_clipped = y.clip(upper=clip_threshold)
    n_clipped = (y > clip_threshold).sum()
    print(f"对 weight 进行异常值裁剪，阈值 ({int(quantile*100)}百分位): {clip_threshold:.2f}，共裁剪 {n_clipped} 个样本")
    return y_clipped, clip_threshold

def create_feature_pipeline(data: pd.DataFrame, feature_order: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    print("开始构建特征流水线...")
    try:
        X = data[feature_order].copy()
        y = data['weight'].copy()
    except KeyError as e: print(f"特征提取失败: 缺少必要的特征列 {e}"); return None, None
    print("  - 步骤 1: 标准化原始特征")
    std_features = [f for f in STD_SCALER_FEATURES if f in X.columns]
    X[std_features] = StandardScaler().fit_transform(X[std_features])
    robust_features = [f for f in ROBUST_SCALER_FEATURES if f in X.columns]
    X[robust_features] = RobustScaler().fit_transform(X[robust_features])
    print("  - 步骤 2: 创造高级特征")
    X['success_rate'] = X['success'] / (X['success'] + X['failure'] + 1e-6)
    X['mb_per_connection'] = (X['upload_mb'] + X['download_mb']) / (X['success'] + 1e-6)
    X['latency_x_connect_time'] = X['latency'] * X['connect_time']
    print(f"特征工程完成，总特征数: {X.shape[1]}")
    X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    return X, y


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    【优化】Optuna 目标函数，以精度为先，但在精度相近时选择复杂度更低的模型。
    """
    param = {
        'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        'n_estimators': trial.suggest_int('n_estimators', *PARAM_GRID['n_estimators'], step=100),
        'learning_rate': trial.suggest_float('learning_rate', *PARAM_GRID['learning_rate']),
        'num_leaves': trial.suggest_int('num_leaves', *PARAM_GRID['num_leaves']),
        'max_depth': trial.suggest_int('max_depth', *PARAM_GRID['max_depth']),
        'min_child_samples': trial.suggest_int('min_child_samples', *PARAM_GRID['min_child_samples']),
        'reg_alpha': trial.suggest_float('reg_alpha', *PARAM_GRID['reg_alpha'], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', *PARAM_GRID['reg_lambda'], log=True),
    }

    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
    r2_scores, rmse_scores = [], []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        
        preds = model.predict(X_test)
        r2_scores.append(r2_score(y_test, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))

    avg_r2 = np.mean(r2_scores)
    avg_rmse = np.mean(rmse_scores)
    
    # 复杂度惩罚项
    complexity_penalty = (param['num_leaves'] * model.best_iteration_) / (PARAM_GRID['num_leaves'][1] * PARAM_GRID['n_estimators'][1])
    
    # 最终的混合分数
    score = (R2_WEIGHT * avg_r2) - (RMSE_WEIGHT * avg_rmse) - (COMPLEXITY_WEIGHT * complexity_penalty)
    
    trial.set_user_attr("avg_r2", avg_r2)
    trial.set_user_attr("avg_rmse", avg_rmse)

    return score

def find_best_params(X: pd.DataFrame, y: pd.Series) -> dict:
    """【第一阶段】执行超参数搜索，返回最佳参数"""
    print(f"【第一阶段】开始使用 Optuna 进行超参数寻优 ({OPTUNA_TRIALS} 次尝试)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print("\n" + "="*25 + " 寻优结果 " + "="*25)
    print(f"寻优完成，最佳混合分数: {study.best_value:.4f}")
    print(f"  - 对应的 R^2: {study.best_trial.user_attrs['avg_r2']:.4f}")
    print(f"  - 对应的 RMSE: {study.best_trial.user_attrs['avg_rmse']:.4f}")
    print("找到的最佳超参数 (已为推理性能优化):")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    return study.best_params

def train_and_save_final_model(X: pd.DataFrame, y: pd.Series, best_params: dict) -> None:
    """【第二阶段】使用最佳参数在全部数据上训练并保存最终的单一模型"""
    print("\n【第二阶段】开始训练最终的单一模型...")
    
    final_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1
    }
    final_params.update(best_params)
    
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X, y)
    
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)
    
    model_path = os.path.join(MODEL_DIR, FINAL_MODEL_NAME)
    
    final_model.booster_.save_model(model_path)
    print(f"最终模型训练完成，已保存至: {model_path}")

# ==============================================================================
# 3. 主程序流程控制 (Main Workflow)
# ==============================================================================

def main() -> None:
    """主程序入口函数"""
    print("=" * 60)
    print("Mihomo/openclash 智能权重模型训练 (CPU版 - 聚焦推理性能)")
    print("=" * 60)
    
    parser = GoTransformParser(GO_FILE)
    feature_order = parser.get_feature_order()
    
    dataset = load_and_clean_data(DATA_FILE)
    if dataset is None: return

    X, y = create_feature_pipeline(dataset, feature_order)
    if X is None: return
    
    y_clipped, _ = clip_weight_outliers(y, OUTLIER_CLIP_QUANTILE)
    
    best_params = find_best_params(X, y_clipped)
    train_and_save_final_model(X, y_clipped, best_params)
    
    print("\n" + "=" * 60)
    print("模型训练流程完成！")
    print(f"单一最终模型已保存至 '{os.path.join(MODEL_DIR, FINAL_MODEL_NAME)}'")
    print("【部署】现在您可以将这个单一模型文件用于您的线上环境。")
    print("e.g. /etc/openclash/Model.bin")
    print("=" * 60)

if __name__ == "__main__":
    main()
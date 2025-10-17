#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遥感影像监督分类系统 - 专业版 v5.1 (PyQt5版本)
=====================================
修复问题:
1. ✅ 修复特征重要性分析中的IndexError
2. ✅ 修复AdaBoost分类器参数错误
3. ✅ 优化特征重要性显示
4. ✅ 增强错误处理机制
5. ✅ 移植完整的分类功能到PyQt5版本
6. ✅ 实现右侧六个标签页的图表绘制功能
"""

import os
import sys
import time
import threading
import queue
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from rasterio import features
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report, 
                           cohen_kappa_score, precision_score, recall_score, f1_score,
                           roc_curve, auc, precision_recall_curve, roc_auc_score)
from sklearn.inspection import permutation_importance
from scipy import ndimage
from skimage import morphology, filters
import warnings
warnings.filterwarnings('ignore')

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, 
                             QCheckBox, QSpinBox, QProgressBar, QTextEdit, QTabWidget, 
                             QScrollArea, QGridLayout, QFrame, QMessageBox, QFileDialog,
                             QRadioButton, QButtonGroup, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

sns.set_style("whitegrid")

# 设置matplotlib中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 检查openpyxl
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("⚠️  未安装openpyxl，将无法导出Excel文件")
    print("   安装: pip install openpyxl")

# ==================== 后端处理类 ====================
class ClassificationBackend:
    """分类处理后端"""
    
    def __init__(self):
        self.RANDOM_STATE = 42
        
        # 预定义颜色
        self.LANDUSE_COLORS = {
            "水体": "lightblue", "河流": "blue", "湖泊": "deepskyblue",
            "植被": "forestgreen", "森林": "darkgreen", "草地": "limegreen",
            "农田": "yellowgreen", "耕地": "olivedrab",
            "建筑": "gray", "城市": "dimgray", "居民地": "slategray",
            "裸地": "tan", "沙地": "wheat", "其他": "darkred"
        }
        
        self.COLOR_PALETTE = ['forestgreen', 'lightblue', 'gray', 'tan', 'yellow', 
                             'darkred', 'purple', 'orange', 'pink', 'brown']
        
        # 检查可选库
        self.check_optional_libraries()
    
    def check_optional_libraries(self):
        """检查可选库是否可用"""
        self.has_xgboost = False
        self.has_lightgbm = False
        
        try:
            import xgboost
            from xgboost import XGBClassifier
            _ = XGBClassifier(n_estimators=10, verbosity=0)
            self.has_xgboost = True
            print("✓ XGBoost 可用")
        except Exception:
            print("✗ XGBoost 不可用")
        
        try:
            import lightgbm
            from lightgbm import LGBMClassifier
            _ = LGBMClassifier(n_estimators=10, verbose=-1)
            self.has_lightgbm = True
            print("✓ LightGBM 可用")
        except Exception:
            print("✗ LightGBM 不可用")
    
    def get_all_classifiers(self, n_estimators=100, fast_mode=False, n_train_samples=None):
        """获取所有可用分类器"""
        if fast_mode:
            n_est = min(50, n_estimators)
            max_depth = 10
            max_iter = 200
        else:
            n_est = n_estimators
            max_depth = 20
            max_iter = 500
        
        if n_train_samples:
            n_components = min(1000, n_train_samples // 2)
        else:
            n_components = 1000
        
        classifiers = {
            "rf": (RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=self.RANDOM_STATE, 
                                         verbose=0, max_depth=max_depth, min_samples_split=5, 
                                         max_features='sqrt'),
                  "随机森林", "Random Forest", False, False, "fast"),
            
            "et": (ExtraTreesClassifier(n_estimators=n_est, n_jobs=-1, random_state=self.RANDOM_STATE,
                                       verbose=0, max_depth=max_depth, min_samples_split=5, max_features='sqrt'),
                  "极端随机树", "Extra Trees", False, False, "fast"),
            
            "dt": (DecisionTreeClassifier(random_state=self.RANDOM_STATE, max_depth=max_depth,
                                         min_samples_split=5, min_samples_leaf=2),
                  "决策树", "Decision Tree", False, False, "very_fast"),
            
            "svm_linear": (SVC(kernel="linear", C=1.0, cache_size=500, probability=True, 
                             random_state=self.RANDOM_STATE, max_iter=max_iter),
                          "SVM-线性核", "SVM Linear", False, True, "medium"),
            
            "linear_svc": (CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=max_iter, random_state=self.RANDOM_STATE,
                                                           dual=False, loss='squared_hinge'), cv=3),
                          "线性SVM(快)", "Linear SVM", False, True, "fast"),
            
            "sgd_svm": (SGDClassifier(loss='hinge', penalty='l2', max_iter=max_iter, n_jobs=-1,
                                     random_state=self.RANDOM_STATE, learning_rate='optimal'),
                       "SGD-SVM", "SGD SVM", False, True, "very_fast"),
            
            "nystroem_svm": (Pipeline([
                ("feature_map", Nystroem(kernel='rbf', gamma=0.1, n_components=n_components, 
                                        random_state=self.RANDOM_STATE)),
                ("sgd", SGDClassifier(max_iter=max_iter, random_state=self.RANDOM_STATE))
            ]), "核近似SVM", "Nystroem SVM", False, True, "fast"),
            
            "rbf_sampler_svm": (Pipeline([
                ("feature_map", RBFSampler(gamma=0.1, n_components=n_components, random_state=self.RANDOM_STATE)),
                ("sgd", SGDClassifier(max_iter=max_iter, random_state=self.RANDOM_STATE))
            ]), "RBF采样SVM", "RBF Sampler", False, True, "fast"),
            
            "svm_rbf": (SVC(kernel="rbf", C=1.0, gamma='scale', cache_size=500, probability=True, 
                          random_state=self.RANDOM_STATE),
                       "SVM-RBF核⚠️", "SVM RBF", False, True, "very_slow"),
            
            "knn": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1, algorithm='ball_tree', leaf_size=30),
                   "K近邻", "KNN", False, True, "slow"),
            
            "nb": (GaussianNB(), "朴素贝叶斯", "Naive Bayes", False, False, "very_fast"),
            
            "gb": (GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.1, max_depth=5,
                                             random_state=self.RANDOM_STATE, verbose=0, subsample=0.8),
                  "梯度提升", "Gradient Boosting", False, False, "medium"),
            
            # 修复AdaBoost参数问题
            "ada": (AdaBoostClassifier(n_estimators=n_est, learning_rate=1.0, 
                                      random_state=self.RANDOM_STATE),
                   "AdaBoost", "AdaBoost", False, False, "medium"),
            
            "lr": (LogisticRegression(max_iter=max_iter, n_jobs=-1, random_state=self.RANDOM_STATE,
                                     verbose=0, solver='lbfgs', multi_class='multinomial'),
                  "逻辑回归", "Logistic Regression", False, True, "very_fast"),
            
            "mlp": (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=max_iter, random_state=self.RANDOM_STATE,
                                 verbose=False, early_stopping=True, validation_fraction=0.1, 
                                 n_iter_no_change=10, learning_rate='adaptive'),
                   "神经网络", "MLP", False, True, "medium"),
        }
        
        if self.has_xgboost:
            try:
                from xgboost import XGBClassifier
                classifiers["xgb"] = (
                    XGBClassifier(n_estimators=n_est, learning_rate=0.1, max_depth=6, n_jobs=-1,
                                 random_state=self.RANDOM_STATE, verbosity=0, tree_method='hist',
                                 subsample=0.8, colsample_bytree=0.8),
                    "XGBoost", "XGBoost", True, False, "fast"
                )
            except Exception:
                pass
        
        if self.has_lightgbm:
            try:
                from lightgbm import LGBMClassifier
                classifiers["lgb"] = (
                    LGBMClassifier(n_estimators=n_est, learning_rate=0.1, max_depth=max_depth, n_jobs=-1,
                                  random_state=self.RANDOM_STATE, verbose=-1, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8, force_col_wise=True),
                    "LightGBM", "LightGBM", False, False, "very_fast"
                )
            except Exception:
                pass
        
        return classifiers

    def get_hyperparameter_grids(self, clf_code, n_samples=None):
        """获取各分类器的超参数网格"""
        grids = {}
        
        # 基础参数
        if n_samples and n_samples < 1000:
            n_estimators_range = [50, 100]
            max_depth_range = [3, 5, 7]
        elif n_samples and n_samples < 10000:
            n_estimators_range = [100, 200]
            max_depth_range = [5, 10, 15]
        else:
            n_estimators_range = [100, 200, 300]
            max_depth_range = [10, 15, 20, None]
        
        if clf_code == "rf":
            grids[clf_code] = {
                'n_estimators': n_estimators_range,
                'max_depth': max_depth_range,
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2', None]
            }
        elif clf_code == "xgb":
            grids[clf_code] = {
                'n_estimators': n_estimators_range,
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        elif clf_code == "lgb":
            grids[clf_code] = {
                'n_estimators': n_estimators_range,
                'max_depth': [3, 6, 9, -1],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 63, 127],
                'subsample': [0.8, 1.0]
            }
        elif clf_code == "svm_linear":
            grids[clf_code] = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto']
            }
        elif clf_code == "svm_rbf":
            grids[clf_code] = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        elif clf_code == "knn":
            grids[clf_code] = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        elif clf_code == "mlp":
            grids[clf_code] = {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        elif clf_code == "gb":
            grids[clf_code] = {
                'n_estimators': n_estimators_range,
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        
        return grids.get(clf_code, {})
    
    def optimize_hyperparameters(self, clf, clf_code, X_train, y_train, cv=3, n_iter=20):
        """超参数优化"""
        param_grid = self.get_hyperparameter_grids(clf_code, len(y_train))
        
        if not param_grid:
            return clf, {}, 0
        
        try:
            # 对于大数据集使用随机搜索，小数据集使用网格搜索
            if len(y_train) > 1000 or len(param_grid) > 10:
                search = RandomizedSearchCV(
                    clf, param_grid, n_iter=min(n_iter, len(param_grid) * 3), 
                    cv=cv, scoring='accuracy', n_jobs=-1, random_state=self.RANDOM_STATE,
                    verbose=0
                )
            else:
                search = GridSearchCV(
                    clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, 
                    verbose=0
                )
            
            search.fit(X_train, y_train)
            
            return search.best_estimator_, search.best_params_, search.best_score_
        
        except Exception as e:
            print(f"超参数优化失败: {e}")
            return clf, {}, 0
    
    def apply_postprocessing(self, classified_array, method='majority', size=3):
        """后处理滤波"""
        if method == 'none':
            return classified_array
        
        # 创建掩膜，只处理非背景区域
        mask = classified_array > 0
        result = classified_array.copy()
        
        if method == 'majority':
            # 多数滤波
            from scipy.ndimage import generic_filter
            
            def majority_filter(x):
                values, counts = np.unique(x, return_counts=True)
                return values[np.argmax(counts)]
            
            result[mask] = generic_filter(
                classified_array, majority_filter, size=size, mode='constant', cval=0
            )[mask]
        
        elif method == 'median':
            # 中值滤波
            result[mask] = ndimage.median_filter(classified_array, size=size)[mask]
        
        elif method == 'opening':
            # 形态学开运算（去除小斑块）
            from skimage.morphology import opening, square
            result = opening(classified_array, square(size))
        
        elif method == 'closing':
            # 形态学闭运算（填充小洞）
            from skimage.morphology import closing, square
            result = closing(classified_array, square(size))
        
        return result
    
    def analyze_feature_importance(self, model, feature_names, X_val, y_val, n_repeats=5):
        """分析特征重要性"""
        try:
            # 对于树模型，使用内置特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # 修复：确保std是正确形状的数组
                if hasattr(model, 'estimators_') and model.estimators_ is not None:
                    try:
                        # 获取所有树的重要性
                        tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
                        std = np.std(tree_importances, axis=0)
                    except:
                        std = np.zeros_like(importances)
                else:
                    std = np.zeros_like(importances)
                
                return {
                    'method': 'builtin',
                    'importances': importances,
                    'std': std,
                    'indices': np.argsort(importances)[::-1]
                }
            
            # 对于其他模型，使用排列重要性
            else:
                try:
                    result = permutation_importance(
                        model, X_val, y_val, n_repeats=n_repeats, 
                        random_state=self.RANDOM_STATE, n_jobs=-1
                    )
                    
                    return {
                        'method': 'permutation',
                        'importances': result.importances_mean,
                        'std': result.importances_std,
                        'indices': np.argsort(result.importances_mean)[::-1]
                    }
                except Exception as e:
                    print(f"排列重要性分析失败: {e}")
                    return None
        
        except Exception as e:
            print(f"特征重要性分析失败: {e}")
            return None
    
    def plot_feature_importance(self, importance_data, feature_names, save_path, clf_name):
        """绘制特征重要性图"""
        if not importance_data:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            indices = importance_data['indices']
            importances = importance_data['importances']
            
            # 修复：正确处理索引和重要性数组
            if len(indices) > 0 and len(importances) > 0:
                # 确保索引不超出范围
                valid_indices = indices[indices < len(importances)]
                sorted_importances = importances[valid_indices]
                
                # 处理标准差
                if 'std' in importance_data and importance_data['std'] is not None:
                    std = importance_data['std']
                    if len(std) == len(importances):
                        sorted_std = std[valid_indices]
                    else:
                        sorted_std = None
                else:
                    sorted_std = None
                
                features = [feature_names[i] for i in valid_indices if i < len(feature_names)]
                
                y_pos = np.arange(len(features))
                
                if sorted_std is not None and len(sorted_std) == len(sorted_importances):
                    ax.barh(y_pos, sorted_importances, xerr=sorted_std, align='center', alpha=0.7)
                else:
                    ax.barh(y_pos, sorted_importances, align='center', alpha=0.7)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('特征重要性')
                ax.set_title(f'{clf_name} - 特征重要性分析')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                return save_path
            else:
                print("特征重要性数据为空或无效")
                return None
                
        except Exception as e:
            print(f"绘制特征重要性图失败: {e}")
            return None
    
    def plot_roc_curves(self, model, X_test, y_test, class_names, save_path, clf_name):
        """绘制ROC曲线（适用于二分类和多分类）"""
        try:
            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)
            else:
                # 对于没有predict_proba的模型，使用决策函数
                y_score = model.decision_function(X_test)
                if y_score.ndim == 1:
                    y_score = y_score.reshape(-1, 1)
            
            n_classes = len(class_names)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if n_classes == 2:
                # 二分类
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC曲线 (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='随机分类器')
                
            else:
                # 多分类 - 一对多
                from sklearn.preprocessing import label_binarize
                
                # 获取所有类别
                classes = np.unique(y_test)
                y_test_bin = label_binarize(y_test, classes=classes)
                
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                # 计算每个类别的ROC曲线
                for i, class_id in enumerate(classes):
                    if y_score.shape[1] > i:  # 确保索引不越界
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                        
                        class_name = class_names.get(class_id, f'类别_{class_id}')
                        ax.plot(fpr[i], tpr[i], lw=2,
                            label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
                
                # 计算宏平均ROC曲线
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
                mean_tpr = np.zeros_like(all_fpr)
                
                for i in range(len(classes)):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                
                mean_tpr /= len(classes)
                
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                
                ax.plot(fpr["macro"], tpr["macro"],
                    label=f'宏平均 (AUC = {roc_auc["macro"]:.2f})',
                    color='navy', linestyle=':', linewidth=4)
                
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                    label='随机分类器', alpha=0.8)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
            ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
            ax.set_title(f'{clf_name} - ROC曲线', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 保存AUC数据
            auc_data = {
                'class_names': class_names,
                'auc_scores': roc_auc if n_classes == 2 else {class_names.get(k, f'类别_{k}'): v for k, v in roc_auc.items()},
                'macro_auc': roc_auc if n_classes == 2 else roc_auc.get("macro", 0)
            }
            
            # 保存详细的ROC数据
            roc_data = {
                'fpr': fpr if n_classes == 2 else {k: v.tolist() for k, v in fpr.items()},
                'tpr': tpr if n_classes == 2 else {k: v.tolist() for k, v in tpr.items()},
                'auc': roc_auc if n_classes == 2 else {k: v for k, v in roc_auc.items()}
            }
            
            with open(save_path.parent / f"{save_path.stem}_auc.json", 'w', encoding='utf-8') as f:
                json.dump(auc_data, f, indent=2, ensure_ascii=False)
            
            with open(save_path.parent / f"{save_path.stem}_roc_data.json", 'w', encoding='utf-8') as f:
                json.dump(roc_data, f, indent=2, ensure_ascii=False)
            
            return save_path, auc_data, roc_data
        
        except Exception as e:
            print(f"ROC曲线绘制失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None, None
    
    def save_model(self, model, file_path):
        """保存训练好的模型"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            return True
        except Exception as e:
            print(f"模型保存失败: {e}")
            return False
    
    def load_model(self, file_path):
        """加载训练好的模型"""
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    
    def get_background_mask(self, image, background_value=0):
        """获取背景掩膜"""
        data = image.values
        if background_value == 0:
            background_mask = np.all(data == 0, axis=0)
        else:
            background_mask = np.all(data == background_value, axis=0)
        return background_mask
    
    def get_shapefile_fields(self, shp_path):
        """获取shapefile的所有字段名"""
        try:
            gdf = gpd.read_file(shp_path)
            return list(gdf.columns)
        except Exception as e:
            print(f"读取shapefile字段失败: {e}")
            return []
    
    def get_class_info_from_shp(self, shp_path, class_attr, name_attr):
        """从shp文件获取类别信息"""
        gdf = gpd.read_file(shp_path)
        
        if name_attr not in gdf.columns or name_attr == class_attr:
            gdf[name_attr] = gdf[class_attr].apply(lambda x: f"类别_{x}")
        
        class_info = gdf[[class_attr, name_attr]].drop_duplicates()
        class_names = dict(zip(class_info[class_attr], class_info[name_attr]))
        
        class_colors = {}
        for i, (class_id, class_name) in enumerate(class_names.items()):
            color_found = False
            for key, color in self.LANDUSE_COLORS.items():
                if key in str(class_name):
                    class_colors[class_id] = color
                    color_found = True
                    break
            if not color_found:
                class_colors[class_id] = self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]
        
        return class_names, class_colors, sorted(class_names.keys())
    
    def rasterize_samples(self, shp, ref_img, attr):
        """矢量栅格化"""
        gdf = gpd.read_file(shp)
        gdf = gdf.to_crs(ref_img.rio.crs)
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attr]))
        
        arr = features.rasterize(shapes=shapes, out_shape=ref_img.shape[1:],
                                transform=ref_img.rio.transform(), fill=0,
                                all_touched=True, dtype="uint16")
        return arr
    
    def extract_samples(self, image, mask, ignore_background=True, background_value=0, max_samples=None):
        """提取样本"""
        data = np.moveaxis(image.values, 0, -1)
        valid = mask > 0
        
        if ignore_background:
            background_mask = self.get_background_mask(image, background_value)
            valid = valid & (~background_mask)
        
        X = data[valid]
        y = mask[valid]
        
        # 清理NaN和Inf
        nan_mask = np.isnan(X).any(axis=1)
        inf_mask = np.isinf(X).any(axis=1)
        bad_mask = nan_mask | inf_mask
        
        n_nan = np.sum(nan_mask)
        n_inf = np.sum(inf_mask)
        
        X = X[~bad_mask]
        y = y[~bad_mask]
        
        # 分层采样
        n_sampled = 0
        if max_samples is not None and len(y) > max_samples:
            n_original = len(y)
            unique_classes = np.unique(y)
            
            if len(unique_classes) > 1:
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, 
                                                 random_state=self.RANDOM_STATE)
                sample_idx, _ = next(splitter.split(X, y))
                X = X[sample_idx]
                y = y[sample_idx]
                n_sampled = n_original - len(y)
            else:
                np.random.seed(self.RANDOM_STATE)
                sample_idx = np.random.choice(len(y), max_samples, replace=False)
                X = X[sample_idx]
                y = y[sample_idx]
                n_sampled = n_original - len(y)
        
        return X, y, n_nan, n_inf, n_sampled
    
    def calculate_metrics(self, y_true, y_pred):
        """计算评价指标"""
        return {
            'overall_accuracy': accuracy_score(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
    
    def save_confusion_matrix(self, y_true, y_pred, class_names, save_path, clf_name):
        """保存混淆矩阵（图片+Excel）"""
        if not HAS_OPENPYXL:
            return None, None
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 保存Excel格式
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        excel_path = save_path.parent / f"{save_path.stem}_confusion_matrix.xlsx"
        cm_df.to_excel(excel_path, engine='openpyxl')
        
        # 绘制并保存图片
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': '样本数量'}, ax=ax)
        
        ax.set_xlabel('预测类别', fontsize=12)
        ax.set_ylabel('真实类别', fontsize=12)
        ax.set_title(f'{clf_name} - 混淆矩阵', fontsize=14, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        img_path = save_path.parent / f"{save_path.stem}_confusion_matrix.png"
        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return excel_path, img_path
    
    def calculate_area_statistics(self, classified_array, pixel_size_x, pixel_size_y, 
                                  class_names, background_value=0):
        """计算面积统计"""
        # 计算每个像元的面积（平方米）
        pixel_area = abs(pixel_size_x * pixel_size_y)
        
        # 统计各类别像元数量
        unique_classes, counts = np.unique(classified_array[classified_array != background_value], 
                                          return_counts=True)
        
        # 计算总像元数（不包括背景）
        total_pixels = np.sum(counts)
        
        # 构建统计表
        stats = []
        for class_id, pixel_count in zip(unique_classes, counts):
            class_name = class_names.get(class_id, f"类别_{class_id}")
            area_m2 = pixel_count * pixel_area
            area_km2 = area_m2 / 1_000_000
            area_ha = area_m2 / 10_000
            percentage = (pixel_count / total_pixels) * 100
            
            stats.append({
                '类别编号': int(class_id),
                '类别名称': class_name,
                '像元数量': int(pixel_count),
                '面积(m²)': area_m2,
                '面积(ha)': area_ha,
                '面积(km²)': area_km2,
                '百分比(%)': percentage
            })
        
        df = pd.DataFrame(stats)
        
        # 添加总计行
        total_row = {
            '类别编号': '',
            '类别名称': '总计',
            '像元数量': int(total_pixels),
            '面积(m²)': df['面积(m²)'].sum(),
            '面积(ha)': df['面积(ha)'].sum(),
            '面积(km²)': df['面积(km²)'].sum(),
            '百分比(%)': 100.0
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        return df
    
    def save_classification_report(self, y_true, y_pred, class_names, save_path):
        """保存详细分类报告（文本+Excel）"""
        if not HAS_OPENPYXL:
            # 仅保存文本格式
            txt_path = save_path.parent / f"{save_path.stem}_classification_report.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(classification_report(y_true, y_pred, 
                                             target_names=class_names,
                                             zero_division=0))
            return None, txt_path
        
        # 获取sklearn的分类报告
        report_dict = classification_report(y_true, y_pred, 
                                           target_names=class_names,
                                           output_dict=True,
                                           zero_division=0)
        
        # 转换为DataFrame
        df = pd.DataFrame(report_dict).transpose()
        
        # 保存Excel
        excel_path = save_path.parent / f"{save_path.stem}_classification_report.xlsx"
        df.to_excel(excel_path, engine='openpyxl')
        
        # 保存文本格式
        txt_path = save_path.parent / f"{save_path.stem}_classification_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(classification_report(y_true, y_pred, 
                                         target_names=class_names,
                                         zero_division=0))
        
        return excel_path, txt_path
    
    def estimate_prediction_time(self, clf_code, n_pixels, speed_tag):
        """估算预测时间"""
        time_per_million = {"very_fast": 1, "fast": 3, "medium": 10, "slow": 30, "very_slow": 300}
        base_time = time_per_million.get(speed_tag, 10)
        return (n_pixels / 1_000_000) * base_time
    
    def predict_by_block(self, model, image, out_path, block_size=512, 
                        ignore_background=True, background_value=0, progress_callback=None,
                        label_encoder=None, scaler=None, postprocessing=None):
        """分块预测"""
        height, width = image.shape[1], image.shape[2]
        prediction = np.zeros((height, width), dtype='uint16')
        
        if ignore_background:
            background_mask = self.get_background_mask(image, background_value)
        
        total_blocks = int(np.ceil(height / block_size))
        
        for i, y in enumerate(range(0, height, block_size)):
            if not hasattr(self, 'is_running') or self.is_running:
                h = min(block_size, height - y)
                block_data = image.isel(y=slice(y, y+h)).values
                data = np.moveaxis(block_data, 0, -1)
                original_shape = data.shape
                data_flat = data.reshape(-1, data.shape[-1])
                
                if ignore_background:
                    block_bg_mask = background_mask[y:y+h, :].flatten()
                    non_bg_indices = ~block_bg_mask
                    
                    if np.any(non_bg_indices):
                        data_to_predict = np.nan_to_num(data_flat[non_bg_indices], 
                                                       nan=0.0, posinf=0.0, neginf=0.0)
                        
                        if scaler is not None:
                            data_to_predict = scaler.transform(data_to_predict)
                        
                        preds_non_bg = model.predict(data_to_predict)
                        
                        if label_encoder is not None:
                            preds_non_bg = label_encoder.inverse_transform(preds_non_bg)
                        
                        preds_flat = np.zeros(len(data_flat), dtype='uint16')
                        preds_flat[non_bg_indices] = preds_non_bg
                        preds = preds_flat.reshape(original_shape[0], original_shape[1])
                    else:
                        preds = np.zeros((original_shape[0], original_shape[1]), dtype='uint16')
                else:
                    data_flat = np.nan_to_num(data_flat, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if scaler is not None:
                        data_flat = scaler.transform(data_flat)
                    
                    preds = model.predict(data_flat)
                    
                    if label_encoder is not None:
                        preds = label_encoder.inverse_transform(preds)
                    
                    preds = preds.reshape(original_shape[0], original_shape[1]).astype("uint16")
                
                prediction[y:y+h, :] = preds
                
                if progress_callback:
                    progress_callback((i + 1) / total_blocks * 100)
            else:
                break
        
        # 应用后处理
        if postprocessing and postprocessing != 'none':
            prediction = self.apply_postprocessing(prediction, method=postprocessing)
        
        # 保存结果
        prediction_da = xr.DataArray(prediction, dims=['y', 'x'],
                                     coords={'y': image.coords['y'], 'x': image.coords['x']})
        
        prediction_da.rio.write_crs(image.rio.crs, inplace=True)
        prediction_da.rio.write_transform(image.rio.transform(), inplace=True)
        prediction_da.rio.write_nodata(background_value, inplace=True)
        
        prediction_da.rio.to_raster(out_path, driver='GTiff', dtype='uint16', 
                                    compress='lzw', tiled=True)
        return out_path


# ==================== PyQt5 GUI 组件 ====================

class MatplotlibWidget(QWidget):
    """Matplotlib图表组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(10, 6)))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.figure = self.canvas.figure
        self.axes = self.figure.add_subplot(111)
    
    def clear(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)


class LogWidget(QTextEdit):
    """日志显示组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        
        # 设置深色主题
        palette = self.palette()
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        self.setPalette(palette)


class FileSelectorWidget(QWidget):
    """文件选择组件"""
    def __init__(self, label_text, file_types="所有文件 (*.*)", is_directory=False, parent=None):
        super().__init__(parent)
        self.is_directory = is_directory
        self.file_types = file_types
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label_text)
        self.entry = QLineEdit()
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse)
        
        layout.addWidget(self.label)
        layout.addWidget(self.entry)
        layout.addWidget(self.browse_btn)
        
        self.setLayout(layout)
    
    def browse(self):
        if self.is_directory:
            directory = QFileDialog.getExistingDirectory(self, "选择目录", self.entry.text())
            if directory:
                self.entry.setText(directory)
        else:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", self.entry.text(), self.file_types)
            if file_path:
                self.entry.setText(file_path)
    
    def get_path(self):
        return self.entry.text()
    
    def set_path(self, path):
        self.entry.setText(path)


class ClassificationThread(QThread):
    """分类线程"""
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, list, object, object, object, object)
    
    def __init__(self, backend, config, selected_classifiers):
        super().__init__()
        self.backend = backend
        self.config = config
        self.selected_classifiers = selected_classifiers
        self.is_running = True
    
    def run(self):
        try:
            # 完整的分类流程
            out_dir = Path(self.config['output_dir'])
            out_dir.mkdir(exist_ok=True)
            
            self.log_signal.emit("="*80)
            self.log_signal.emit("  遥感影像监督分类系统 v5.1 - 专业版 (PyQt5)")
            self.log_signal.emit("="*80)
            self.log_signal.emit(f"选择的分类器: {len(self.selected_classifiers)} 个")
            self.log_signal.emit(f"背景值: {self.config['background_value']}")
            self.log_signal.emit("")
            
            # 读取影像
            self.log_signal.emit("📁 读取影像...")
            self.status_signal.emit("读取影像...")
            img = rxr.open_rasterio(self.config['image_path'], masked=True)
            n_pixels = img.shape[1] * img.shape[2]
            self.log_signal.emit(f"   尺寸: {img.shape[1]}×{img.shape[2]} = {n_pixels:,} 像元")
            
            # 获取像元大小
            transform = img.rio.transform()
            pixel_size_x = transform[0]  # X方向分辨率
            pixel_size_y = abs(transform[4])  # Y方向分辨率
            self.log_signal.emit(f"   分辨率: {pixel_size_x:.2f} × {pixel_size_y:.2f} 米")
            
            # 获取波段名称
            band_names = [f"波段_{i+1}" for i in range(img.shape[0])]
            if hasattr(img, 'long_name') and img.long_name:
                band_names = img.long_name
            elif hasattr(img, 'band') and img.band.values is not None:
                band_names = [f"波段_{b}" for b in img.band.values]
            
            self.log_signal.emit(f"   波段数: {len(band_names)}")
            
            if not self.is_running:
                return
            
            # 读取类别信息
            self.log_signal.emit("📊 读取类别信息...")
            class_names, class_colors, _ = self.backend.get_class_info_from_shp(
                self.config['train_shp_path'], 
                self.config['class_attr'], 
                self.config['name_attr']
            )
            self.log_signal.emit(f"   类别: {list(class_names.values())}")
            
            # 提取训练样本
            self.log_signal.emit("🎯 处理训练样本...")
            self.status_signal.emit("处理训练样本...")
            train_mask = self.backend.rasterize_samples(
                self.config['train_shp_path'], img, self.config['class_attr']
            )
            
            max_samples = self.config['max_samples'] if self.config['enable_sampling'] else None
            
            X_train, y_train, n_nan, n_inf, n_sampled = self.backend.extract_samples(
                img, train_mask, 
                ignore_background=self.config['ignore_background'],
                background_value=self.config['background_value'],
                max_samples=max_samples
            )
            
            self.log_signal.emit(f"   训练样本数: {len(y_train):,}")
            if n_nan > 0:
                self.log_signal.emit(f"   └─ 移除NaN: {n_nan:,}")
            if n_sampled > 0:
                self.log_signal.emit(f"   └─ 采样减少: {n_sampled:,}")
            
            if not self.is_running:
                return
            
            # 提取验证样本
            val_exists = os.path.exists(self.config['val_shp_path'])
            X_val, yv_true = None, None
            valid_val = None
            
            if val_exists:
                self.log_signal.emit("✅ 处理验证样本...")
                val_mask = self.backend.rasterize_samples(
                    self.config['val_shp_path'], img, self.config['class_attr']
                )
                
                if self.config['ignore_background']:
                    background_mask = self.backend.get_background_mask(
                        img, self.config['background_value']
                    )
                    valid_val = (val_mask > 0) & (~background_mask)
                else:
                    valid_val = val_mask > 0
                
                # 提取验证样本特征
                data = np.moveaxis(img.values, 0, -1)
                X_val = data[valid_val]
                yv_true = val_mask[valid_val]
                
                # 清理NaN和Inf
                nan_mask = np.isnan(X_val).any(axis=1)
                inf_mask = np.isinf(X_val).any(axis=1)
                bad_mask = nan_mask | inf_mask
                X_val = X_val[~bad_mask]
                yv_true = yv_true[~bad_mask]
                
                self.log_signal.emit(f"   验证样本数: {len(yv_true):,}")
            
            # 分类器训练和评估
            all_classifiers = self.backend.get_all_classifiers(
                self.config['n_estimators'], 
                fast_mode=self.config['fast_mode'],
                n_train_samples=len(y_train)
            )
            
            comparison_results = []
            total_start_time = time.time()
            best_accuracy = 0
            best_clf_code = None
            best_y_true = None
            best_y_pred = None
            best_class_names = class_names
            best_class_colors = class_colors
            best_result_path = None
            
            # 存储特征重要性数据和ROC数据
            feature_importance_data = {}
            roc_data_dict = {}
            
            for i, clf_code in enumerate(self.selected_classifiers):
                if not self.is_running:
                    break
                
                clf, clf_name, clf_desc, needs_encoding, needs_scaling, speed_tag = all_classifiers[clf_code]
                
                self.log_signal.emit(f"\n{'='*80}")
                self.log_signal.emit(f"[{i+1}/{len(self.selected_classifiers)}] {clf_name}")
                self.log_signal.emit(f"{'='*80}")
                
                self.status_signal.emit(f"[{i+1}/{len(self.selected_classifiers)}] 训练 {clf_name}...")
                
                clf_dir = out_dir / clf_code
                clf_dir.mkdir(exist_ok=True)
                
                try:
                    # 数据预处理
                    label_encoder = None
                    scaler = None
                    X_train_use = X_train.copy()
                    y_train_use = y_train.copy()
                    
                    if needs_encoding:
                        self.log_signal.emit("   🔄 标签编码...")
                        label_encoder = LabelEncoder()
                        y_train_use = label_encoder.fit_transform(y_train)
                    
                    if needs_scaling:
                        self.log_signal.emit("   📏 特征缩放...")
                        scaler = StandardScaler()
                        X_train_use = scaler.fit_transform(X_train_use)
                    
                    # 超参数优化
                    best_params = {}
                    optimization_score = 0
                    
                    if self.config['enable_hyperparameter_optimization']:
                        self.log_signal.emit("   🎯 超参数优化中...")
                        opt_start = time.time()
                        
                        clf_optimized, best_params, optimization_score = self.backend.optimize_hyperparameters(
                            clf, clf_code, X_train_use, y_train_use,
                            n_iter=self.config['hyperparameter_iterations']
                        )
                        
                        if clf_optimized is not clf:
                            clf = clf_optimized
                            self.log_signal.emit(f"   ✓ 超参数优化完成: {time.time() - opt_start:.2f}秒")
                            self.log_signal.emit(f"   📊 优化后得分: {optimization_score:.4f}")
                            if best_params:
                                self.log_signal.emit(f"   ⚙️  最佳参数: {best_params}")
                        else:
                            self.log_signal.emit("   ⚠️  超参数优化未改善性能，使用默认参数")
                    
                    # 训练
                    self.log_signal.emit("   🔨 训练中...")
                    train_start = time.time()
                    clf.fit(X_train_use, y_train_use)
                    train_time = time.time() - train_start
                    self.log_signal.emit(f"   ✓ 训练完成: {train_time:.2f}秒")
                    
                    # 保存模型
                    if self.config['enable_model_saving']:
                        model_path = clf_dir / f"{clf_code}_model.pkl"
                        if self.backend.save_model(clf, model_path):
                            self.log_signal.emit(f"   💾 模型已保存: {model_path.name}")
                    
                    # 训练集精度
                    y_train_pred = clf.predict(X_train_use)
                    
                    if label_encoder is not None:
                        y_train_pred = label_encoder.inverse_transform(y_train_pred)
                    
                    train_metrics = self.backend.calculate_metrics(y_train, y_train_pred)
                    self.log_signal.emit(f"   📈 训练集 - 精度: {train_metrics['overall_accuracy']:.4f}")
                    
                    # 保存训练集混淆矩阵和报告
                    if HAS_OPENPYXL:
                        self.log_signal.emit("   💾 保存训练集结果...")
                        train_classes = sorted(np.unique(y_train))
                        train_class_names = [class_names.get(c, f'类别_{c}') for c in train_classes]
                        
                        excel_path, img_path = self.backend.save_confusion_matrix(
                            y_train, y_train_pred, train_class_names,
                            clf_dir / f"{clf_code}_train",
                            f"{clf_name} (训练集)"
                        )
                        if excel_path:
                            self.log_signal.emit(f"      ✓ 混淆矩阵: {excel_path.name}")
                            self.log_signal.emit(f"      ✓ 图片: {img_path.name}")
                        
                        excel_path, txt_path = self.backend.save_classification_report(
                            y_train, y_train_pred, train_class_names,
                            clf_dir / f"{clf_code}_train"
                        )
                        if excel_path:
                            self.log_signal.emit(f"      ✓ 分类报告: {excel_path.name}")
                    
                    # 特征重要性分析
                    importance_data = None
                    if self.config['enable_feature_importance'] and X_val is not None:
                        self.log_signal.emit("   📊 分析特征重要性...")
                        importance_data = self.backend.analyze_feature_importance(
                            clf, band_names, X_val, yv_true
                        )
                        
                        if importance_data:
                            feature_importance_data[clf_code] = importance_data
                            importance_path = clf_dir / f"{clf_code}_feature_importance.png"
                            saved_path = self.backend.plot_feature_importance(
                                importance_data, band_names, importance_path, clf_name
                            )
                            if saved_path:
                                self.log_signal.emit(f"      ✓ 特征重要性图: {saved_path.name}")
                            else:
                                self.log_signal.emit("      ⚠️  特征重要性图生成失败")
                        else:
                            self.log_signal.emit("      ⚠️  特征重要性分析失败")
                    
                    # ROC曲线分析
                    roc_data = None
                    if self.config['enable_roc_analysis'] and X_val is not None:
                        self.log_signal.emit("   📉 绘制ROC曲线...")
                        val_classes = sorted(np.unique(yv_true))
                        val_class_names = [class_names.get(c, f'类别_{c}') for c in val_classes]
                        
                        roc_path = clf_dir / f"{clf_code}_roc_curve.png"
                        roc_path, auc_data, roc_data = self.backend.plot_roc_curves(
                            clf, X_val, yv_true, class_names, roc_path, clf_name
                        )
                        
                        if roc_path:
                            roc_data_dict[clf_code] = roc_data
                            self.log_signal.emit(f"      ✓ ROC曲线: {roc_path.name}")
                            if auc_data:
                                self.log_signal.emit(f"      ✓ 平均AUC: {auc_data['macro_auc']:.4f}")
                        else:
                            self.log_signal.emit("      ⚠️ ROC曲线绘制失败")
                    
                    if not self.is_running:
                        break
                    
                    # 预测整幅影像
                    self.log_signal.emit("   🗺️  预测影像...")
                    self.status_signal.emit(f"[{i+1}/{len(self.selected_classifiers)}] 预测 {clf_name}...")
                    
                    pred_start = time.time()
                    classified_path = clf_dir / f"classified_{clf_code}.tif"
                    
                    # 后处理设置
                    postprocessing = self.config['postprocessing_method'] if self.config['enable_postprocessing'] else 'none'
                    
                    # 传递运行状态给后端
                    self.backend.is_running = self.is_running
                    
                    result_path = self.backend.predict_by_block(
                        clf, img, classified_path, 
                        block_size=self.config['block_size'],
                        ignore_background=self.config['ignore_background'],
                        background_value=self.config['background_value'],
                        progress_callback=lambda p: self.progress_signal.emit(p),
                        label_encoder=label_encoder,
                        scaler=scaler,
                        postprocessing=postprocessing
                    )
                    
                    if result_path and os.path.exists(result_path):
                        pred_time = time.time() - pred_start
                        self.log_signal.emit(f"   ✓ 预测完成: {pred_time:.2f}秒")
                        
                        # 读取分类结果进行面积统计
                        self.log_signal.emit("   📊 计算面积统计...")
                        with rxr.open_rasterio(classified_path) as pred_img:
                            pred_arr = pred_img.values.squeeze()
                        
                        # 计算面积统计
                        area_stats = self.backend.calculate_area_statistics(
                            pred_arr, pixel_size_x, pixel_size_y,
                            class_names, self.config['background_value']
                        )
                        
                        # 保存面积统计
                        if HAS_OPENPYXL:
                            area_excel_path = clf_dir / f"{clf_code}_area_statistics.xlsx"
                            area_stats.to_excel(area_excel_path, index=False, engine='openpyxl')
                            self.log_signal.emit(f"      ✓ 面积统计: {area_excel_path.name}")
                        
                        # 显示面积统计摘要
                        self.log_signal.emit("   面积统计摘要:")
                        for _, row in area_stats.iterrows():
                            if row['类别名称'] == '总计':
                                self.log_signal.emit(f"      ─────────────────────")
                                self.log_signal.emit(f"      {row['类别名称']}: {row['面积(km²)']:.2f} km²")
                            else:
                                self.log_signal.emit(f"      {row['类别名称']}: {row['面积(km²)']:.2f} km² ({row['百分比(%)']:.2f}%)")
                        
                        # 验证集精度
                        val_metrics = {'overall_accuracy': np.nan, 'kappa': np.nan}
                        yv_pred = None
                        
                        if val_exists:
                            yv_pred = pred_arr[valid_val]
                            val_metrics = self.backend.calculate_metrics(yv_true, yv_pred)
                            self.log_signal.emit(f"   📊 验证集 - 精度: {val_metrics['overall_accuracy']:.4f}")
                            
                            # 保存验证集混淆矩阵和报告
                            if HAS_OPENPYXL:
                                self.log_signal.emit("   💾 保存验证集结果...")
                                val_classes = sorted(np.unique(yv_true))
                                val_class_names = [class_names.get(c, f'类别_{c}') for c in val_classes]
                                
                                excel_path, img_path = self.backend.save_confusion_matrix(
                                    yv_true, yv_pred, val_class_names,
                                    clf_dir / f"{clf_code}_val",
                                    f"{clf_name} (验证集)"
                                )
                                if excel_path:
                                    self.log_signal.emit(f"      ✓ 混淆矩阵: {excel_path.name}")
                                    self.log_signal.emit(f"      ✓ 图片: {img_path.name}")
                                
                                excel_path, txt_path = self.backend.save_classification_report(
                                    yv_true, yv_pred, val_class_names,
                                    clf_dir / f"{clf_code}_val"
                                )
                                if excel_path:
                                    self.log_signal.emit(f"      ✓ 分类报告: {excel_path.name}")
                            
                            # 记录最佳分类器
                            if val_metrics['overall_accuracy'] > best_accuracy:
                                best_accuracy = val_metrics['overall_accuracy']
                                best_clf_code = clf_code
                                best_y_true = yv_true
                                best_y_pred = yv_pred
                                best_result_path = classified_path
                        
                        # 记录结果
                        result = {
                            '分类器代码': clf_code,
                            '分类器名称': clf_name,
                            '训练集精度': train_metrics['overall_accuracy'],
                            '训练集Kappa': train_metrics['kappa'],
                            '验证集精度': val_metrics['overall_accuracy'],
                            '验证集Kappa': val_metrics['kappa'],
                            '训练时间(秒)': train_time,
                            '预测时间(秒)': pred_time,
                            '超参数优化': bool(best_params),
                            '优化得分': optimization_score,
                        }
                        comparison_results.append(result)
                        
                        self.log_signal.emit(f"   ✅ {clf_name} 完成!")
                    else:
                        self.log_signal.emit(f"   ❌ {clf_name} 预测失败或用户取消")
                    
                except Exception as e:
                    self.log_signal.emit(f"   ❌ {clf_name} 失败: {str(e)}")
                    import traceback
                    self.log_signal.emit(traceback.format_exc())
                    continue
                
                self.progress_signal.emit((i + 1) / len(self.selected_classifiers) * 100)
            
            # 生成报告
            if comparison_results and self.is_running:
                total_time = time.time() - total_start_time
                
                self.log_signal.emit(f"\n{'='*80}")
                self.log_signal.emit("📝 生成总体报告...")
                
                comparison_df = pd.DataFrame(comparison_results)
                
                # 导出Excel
                if HAS_OPENPYXL:
                    excel_path = out_dir / "classification_comparison.xlsx"
                    comparison_df.to_excel(excel_path, index=False, engine='openpyxl')
                    self.log_signal.emit(f"✓ Excel报告已保存: {excel_path}")
                
                # 文字报告
                with open(out_dir / "comparison_summary.txt", 'w', encoding='utf-8') as f:
                    f.write("遥感影像分类器性能对比报告\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"影像: {img.shape[1]}×{img.shape[2]}\n")
                    f.write(f"分辨率: {pixel_size_x:.2f}m × {pixel_size_y:.2f}m\n")
                    f.write(f"训练样本: {len(y_train):,}\n")
                    f.write(f"成功: {len(comparison_results)}/{len(self.selected_classifiers)}\n")
                    f.write(f"总耗时: {total_time/60:.1f} 分钟\n\n")
                    
                    sorted_df = comparison_df.sort_values('验证集精度', ascending=False)
                    f.write("验证集精度排名:\n")
                    f.write("-"*70 + "\n")
                    for idx, (_, row) in enumerate(sorted_df.iterrows(), 1):
                        f.write(f"{idx}. {row['分类器名称']:15s} - "
                               f"精度: {row['验证集精度']:.4f}, "
                               f"Kappa: {row['验证集Kappa']:.4f}\n")
                
                self.log_signal.emit("\n✅ 所有任务完成!")
                self.log_signal.emit(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
                
                best_clf = comparison_df.loc[comparison_df['验证集精度'].idxmax()]
                self.log_signal.emit(f"\n🏆 最佳: {best_clf['分类器名称']} ({best_clf['验证集精度']:.4f})")
                
                self.status_signal.emit(f"✅ 完成! 最佳: {best_clf['分类器名称']}")
                
                # 发射完成信号，包含所有需要的数据
                self.finished_signal.emit(
                    True, 
                    f"🎉 分类任务完成!\n\n✅ 成功: {len(comparison_results)}/{len(self.selected_classifiers)}\n🏆 最佳: {best_clf['分类器名称']} ({best_clf['验证集精度']:.4f})",
                    comparison_results,
                    (best_y_true, best_y_pred, best_class_names) if best_y_true is not None else None,
                    feature_importance_data,
                    roc_data_dict,
                    (self.config['image_path'], best_result_path, best_class_names, best_class_colors) if best_result_path else None
                )
            else:
                self.finished_signal.emit(False, "分类任务被取消或没有成功完成", [], None, {}, {}, None)
                
        except Exception as e:
            self.log_signal.emit(f"\n❌ 错误: {str(e)}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, f"发生错误:\n{str(e)}", [], None, {}, {}, None)
    
    def stop(self):
        self.is_running = False


# ==================== 主窗口 ====================

class ClassificationGUI(QMainWindow):
    """遥感影像分类GUI主界面（PyQt5版本）"""
    
    def __init__(self):
        super().__init__()
        self.backend = ClassificationBackend()
        self.init_data()  # 先初始化数据
        self.init_ui()    # 再初始化界面
    
    def init_data(self):
        """初始化数据"""
        self.image_path = ""
        self.train_shp_path = ""
        self.val_shp_path = ""
        self.output_dir = str(Path("./results_gui"))
        
        self.train_fields = []
        self.class_attr = ""
        self.name_attr = ""
        
        # 分类器选择
        self.classifier_vars = {}
        all_classifiers = self.backend.get_all_classifiers()
        for code in all_classifiers.keys():
            self.classifier_vars[code] = False
        
        # 运行状态
        self.is_running = False
        self.classification_thread = None
        
        # 存储结果数据
        self.comparison_results = []
        self.current_confusion_matrix = None
        self.current_y_true = None
        self.current_y_pred = None
        self.class_names_dict = {}
        self.class_colors_dict = {}
        self.best_result_path = None
        
        # 存储图表数据
        self.feature_importance_data = {}
        self.roc_data_dict = {}
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("遥感影像监督分类系统 v1.0 - 专业版 (PyQt5)")
        self.setGeometry(100, 50, 1600, 900)
        
        # 设置窗口最大化
        self.showMaximized()
        
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板 - 参数设置
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧面板 - 结果显示
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例 - 右侧更大，适合图表显示
        splitter.setSizes([400, 1200])  # 左侧400，右侧1200
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def create_left_panel(self):
        """创建左侧参数面板"""
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 1. 数据文件
        file_group = QGroupBox("📁 数据文件")
        file_layout = QVBoxLayout()
        
        self.image_selector = FileSelectorWidget("影像文件:", "GeoTIFF (*.tif *.tiff)")
        self.train_shp_selector = FileSelectorWidget("训练样本:", "Shapefile (*.shp)")
        self.val_shp_selector = FileSelectorWidget("验证样本:", "Shapefile (*.shp)")
        self.output_selector = FileSelectorWidget("输出目录:", is_directory=True)
        self.output_selector.set_path(self.output_dir)  # 使用set_path方法
        
        file_layout.addWidget(self.image_selector)
        file_layout.addWidget(self.train_shp_selector)
        file_layout.addWidget(self.val_shp_selector)
        file_layout.addWidget(self.output_selector)
        
        file_group.setLayout(file_layout)
        scroll_layout.addWidget(file_group)
        
        # 2. 字段配置
        field_group = QGroupBox("🏷️ 字段配置")
        field_layout = QGridLayout()
        
        field_layout.addWidget(QLabel("类别编号字段:"), 0, 0)
        self.class_attr_combo = QComboBox()
        field_layout.addWidget(self.class_attr_combo, 0, 1)
        
        field_layout.addWidget(QLabel("类别名称字段:"), 1, 0)
        self.name_attr_combo = QComboBox()
        field_layout.addWidget(self.name_attr_combo, 1, 1)
        
        refresh_btn = QPushButton("🔄 刷新字段列表")
        refresh_btn.clicked.connect(self.refresh_fields)
        field_layout.addWidget(refresh_btn, 0, 2, 2, 1)
        
        field_group.setLayout(field_layout)
        scroll_layout.addWidget(field_group)
        
        # 3. 背景值设置
        bg_group = QGroupBox("🎨 背景值设置")
        bg_layout = QGridLayout()
        
        self.ignore_background = QCheckBox("忽略背景值")
        self.ignore_background.setChecked(True)
        bg_layout.addWidget(self.ignore_background, 0, 0)
        
        bg_layout.addWidget(QLabel("背景值:"), 1, 0)
        self.background_value = QSpinBox()
        self.background_value.setRange(-9999, 9999)
        self.background_value.setValue(0)
        bg_layout.addWidget(self.background_value, 1, 1)
        
        bg_layout.addWidget(QLabel("(默认0, 常见: -9999, 255)"), 1, 2)
        
        bg_group.setLayout(bg_layout)
        scroll_layout.addWidget(bg_group)
        
        # 4. 分类参数
        param_group = QGroupBox("⚙️ 分类参数")
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("树模型数量:"), 0, 0)
        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(10, 500)
        self.n_estimators.setValue(100)
        param_layout.addWidget(self.n_estimators, 0, 1)
        
        param_layout.addWidget(QLabel("分块大小:"), 1, 0)
        self.block_size = QSpinBox()
        self.block_size.setRange(256, 2048)
        self.block_size.setSingleStep(256)
        self.block_size.setValue(512)
        param_layout.addWidget(self.block_size, 1, 1)
        
        param_group.setLayout(param_layout)
        scroll_layout.addWidget(param_group)
        
        # 5. 性能优化
        opt_group = QGroupBox("⚡ 性能优化")
        opt_layout = QVBoxLayout()
        
        sample_layout = QHBoxLayout()
        self.enable_sampling = QCheckBox("启用采样")
        self.enable_sampling.setChecked(True)
        self.enable_sampling.toggled.connect(self.toggle_sampling)
        sample_layout.addWidget(self.enable_sampling)
        
        sample_layout.addWidget(QLabel("最大样本数:"))
        self.max_samples = QSpinBox()
        self.max_samples.setRange(10000, 200000)
        self.max_samples.setSingleStep(10000)
        self.max_samples.setValue(50000)
        sample_layout.addWidget(self.max_samples)
        
        opt_layout.addLayout(sample_layout)
        
        self.fast_mode = QCheckBox("快速模式")
        opt_layout.addWidget(self.fast_mode)
        
        opt_group.setLayout(opt_layout)
        scroll_layout.addWidget(opt_group)
        
        # 6. 高级功能
        advanced_group = QGroupBox("🚀 高级功能")
        advanced_layout = QVBoxLayout()
        
        # 超参数优化
        hyper_layout = QHBoxLayout()
        self.enable_hyperparameter_optimization = QCheckBox("超参数优化")
        hyper_layout.addWidget(self.enable_hyperparameter_optimization)
        
        hyper_layout.addWidget(QLabel("方法:"))
        self.hyperparameter_optimization_method = QComboBox()
        self.hyperparameter_optimization_method.addItems(["random", "grid"])
        hyper_layout.addWidget(self.hyperparameter_optimization_method)
        
        hyper_layout.addWidget(QLabel("迭代次数:"))
        self.hyperparameter_iterations = QSpinBox()
        self.hyperparameter_iterations.setRange(5, 100)
        self.hyperparameter_iterations.setValue(20)
        hyper_layout.addWidget(self.hyperparameter_iterations)
        
        advanced_layout.addLayout(hyper_layout)
        
        # 后处理
        post_layout = QHBoxLayout()
        self.enable_postprocessing = QCheckBox("后处理滤波")
        self.enable_postprocessing.setChecked(True)
        post_layout.addWidget(self.enable_postprocessing)
        
        post_layout.addWidget(QLabel("方法:"))
        self.postprocessing_method = QComboBox()
        self.postprocessing_method.addItems(["none", "majority", "median", "opening", "closing"])
        post_layout.addWidget(self.postprocessing_method)
        
        post_layout.addWidget(QLabel("窗口大小:"))
        self.postprocessing_size = QSpinBox()
        self.postprocessing_size.setRange(3, 9)
        self.postprocessing_size.setSingleStep(2)
        self.postprocessing_size.setValue(3)
        post_layout.addWidget(self.postprocessing_size)
        
        advanced_layout.addLayout(post_layout)
        
        # 分析选项
        analysis_layout = QHBoxLayout()
        self.enable_feature_importance = QCheckBox("特征重要性分析")
        self.enable_feature_importance.setChecked(True)
        analysis_layout.addWidget(self.enable_feature_importance)
        
        self.enable_roc_analysis = QCheckBox("ROC曲线分析")
        self.enable_roc_analysis.setChecked(True)
        analysis_layout.addWidget(self.enable_roc_analysis)
        
        self.enable_model_saving = QCheckBox("保存模型")
        self.enable_model_saving.setChecked(True)
        analysis_layout.addWidget(self.enable_model_saving)
        
        advanced_layout.addLayout(analysis_layout)
        
        advanced_group.setLayout(advanced_layout)
        scroll_layout.addWidget(advanced_group)
        
        # 7. 分类器选择
        clf_group = QGroupBox("🤖 分类器选择")
        clf_layout = QVBoxLayout()
        
        # 快捷按钮
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_classifiers)
        btn_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.clicked.connect(self.deselect_all_classifiers)
        btn_layout.addWidget(deselect_all_btn)
        
        recommended_btn = QPushButton("✓推荐")
        recommended_btn.clicked.connect(self.select_recommended)
        btn_layout.addWidget(recommended_btn)
        
        fast_btn = QPushButton("⚡快速")
        fast_btn.clicked.connect(self.select_fast)
        btn_layout.addWidget(fast_btn)
        
        clf_layout.addLayout(btn_layout)
        
        # 分类器复选框
        clf_scroll = QScrollArea()
        clf_scroll_widget = QWidget()
        clf_scroll_layout = QGridLayout(clf_scroll_widget)
        
        all_classifiers = self.backend.get_all_classifiers()
        
        # SVM组
        row = 0
        clf_scroll_layout.addWidget(QLabel("SVM:"), row, 0)
        row += 1
        
        svm_codes = ["svm_linear", "linear_svc", "sgd_svm", "nystroem_svm", 
                     "rbf_sampler_svm", "svm_rbf"]
        for code in svm_codes:
            if code in all_classifiers:
                _, name, _, _, _, _ = all_classifiers[code]
                checkbox = QCheckBox(name)
                checkbox.setChecked(False)
                self.classifier_vars[code] = checkbox
                clf_scroll_layout.addWidget(checkbox, row, 0)
                row += 1
        
        # 树模型
        clf_scroll_layout.addWidget(QLabel("树模型:"), row, 0)
        row += 1
        
        tree_codes = ["rf", "et", "dt", "xgb", "lgb", "gb", "ada"]
        for code in tree_codes:
            if code in all_classifiers:
                _, name, _, _, _, _ = all_classifiers[code]
                checkbox = QCheckBox(name)
                checkbox.setChecked(False)
                self.classifier_vars[code] = checkbox
                clf_scroll_layout.addWidget(checkbox, row, 0)
                row += 1
        
        # 其他
        clf_scroll_layout.addWidget(QLabel("其他:"), row, 0)
        row += 1
        
        other_codes = ["knn", "nb", "lr", "mlp"]
        for code in other_codes:
            if code in all_classifiers:
                _, name, _, _, _, _ = all_classifiers[code]
                checkbox = QCheckBox(name)
                checkbox.setChecked(False)
                self.classifier_vars[code] = checkbox
                clf_scroll_layout.addWidget(checkbox, row, 0)
                row += 1
        
        clf_scroll.setWidget(clf_scroll_widget)
        clf_scroll.setFixedHeight(200)
        clf_layout.addWidget(clf_scroll)
        
        clf_group.setLayout(clf_layout)
        scroll_layout.addWidget(clf_group)
        
        # 8. 运行控制
        control_group = QGroupBox("🎮 运行控制")
        control_layout = QVBoxLayout()
        
        # 控制按钮
        btn_control_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始分类")
        self.start_btn.clicked.connect(self.start_classification)
        btn_control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏸ 停止")
        self.stop_btn.clicked.connect(self.stop_classification)
        self.stop_btn.setEnabled(False)
        btn_control_layout.addWidget(self.stop_btn)
        
        save_config_btn = QPushButton("💾 保存配置")
        save_config_btn.clicked.connect(self.save_config)
        btn_control_layout.addWidget(save_config_btn)
        
        load_config_btn = QPushButton("📂 加载配置")
        load_config_btn.clicked.connect(self.load_config)
        btn_control_layout.addWidget(load_config_btn)
        
        open_result_btn = QPushButton("📁 打开结果")
        open_result_btn.clicked.connect(self.open_result_dir)
        btn_control_layout.addWidget(open_result_btn)
        
        control_layout.addLayout(btn_control_layout)
        
        # 进度条
        control_layout.addWidget(QLabel("进度:"))
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        # 状态
        self.status_label = QLabel("就绪")
        control_layout.addWidget(self.status_label)
        
        control_group.setLayout(control_layout)
        scroll_layout.addWidget(control_group)
        
        scroll_area.setWidget(scroll_widget)
        return scroll_area
    
    def create_right_panel(self):
        """创建右侧结果显示面板"""
        self.tab_widget = QTabWidget()
        
        # 标签页1：运行日志
        self.log_widget = LogWidget()
        self.tab_widget.addTab(self.log_widget, "📝 运行日志")
        
        # 标签页2：精度对比
        self.accuracy_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.accuracy_widget, "📊 精度对比")
        
        # 标签页3：混淆矩阵
        self.cm_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.cm_widget, "🔥 混淆矩阵")
        
        # 标签页4：时间对比
        self.time_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.time_widget, "⏱️ 时间对比")
        
        # 标签页5：分类结果预览
        self.result_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.result_widget, "🗺️ 结果预览")
        
        # 标签页6：特征重要性
        self.feature_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.feature_widget, "📈 特征重要性")
        
        # 标签页7：ROC曲线
        self.roc_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.roc_widget, "📉 ROC曲线")
        
        return self.tab_widget
    
    def toggle_sampling(self):
        """切换采样状态"""
        self.max_samples.setEnabled(self.enable_sampling.isChecked())
    
    def refresh_fields(self):
        """刷新字段列表"""
        train_shp = self.train_shp_selector.get_path()
        if not train_shp or not os.path.exists(train_shp):
            QMessageBox.warning(self, "警告", "请先选择训练样本文件！")
            return
        
        fields = self.backend.get_shapefile_fields(train_shp)
        if fields:
            fields = [f for f in fields if f.lower() != 'geometry']
            
            self.class_attr_combo.clear()
            self.name_attr_combo.clear()
            
            self.class_attr_combo.addItems(fields)
            self.name_attr_combo.addItems(fields)
            
            if 'class' in fields:
                self.class_attr_combo.setCurrentText('class')
            elif 'Class' in fields:
                self.class_attr_combo.setCurrentText('Class')
            elif fields:
                self.class_attr_combo.setCurrentText(fields[0])
            
            if 'name' in fields:
                self.name_attr_combo.setCurrentText('name')
            elif 'Name' in fields:
                self.name_attr_combo.setCurrentText('Name')
            elif len(fields) > 1:
                self.name_attr_combo.setCurrentText(fields[1])
            elif fields:
                self.name_attr_combo.setCurrentText(fields[0])
            
            QMessageBox.information(self, "成功", f"已加载 {len(fields)} 个字段")
        else:
            QMessageBox.critical(self, "错误", "无法读取字段列表！")
    
    def select_all_classifiers(self):
        """全选分类器"""
        for checkbox in self.classifier_vars.values():
            checkbox.setChecked(True)
    
    def deselect_all_classifiers(self):
        """全不选分类器"""
        for checkbox in self.classifier_vars.values():
            checkbox.setChecked(False)
    
    def select_recommended(self):
        """选择推荐分类器"""
        recommended = ["rf", "xgb", "et", "lgb", "linear_svc", "nystroem_svm"]
        for code, checkbox in self.classifier_vars.items():
            checkbox.setChecked(code in recommended)
    
    def select_fast(self):
        """选择快速分类器"""
        fast = ["rf", "et", "dt", "xgb", "lgb", "nb", "lr", "sgd_svm", "linear_svc"]
        for code, checkbox in self.classifier_vars.items():
            checkbox.setChecked(code in fast)
    
    def save_config(self):
        """保存配置"""
        config = {
            'image_path': self.image_selector.get_path(),
            'train_shp_path': self.train_shp_selector.get_path(),
            'val_shp_path': self.val_shp_selector.get_path(),
            'output_dir': self.output_selector.get_path(),
            'class_attr': self.class_attr_combo.currentText(),
            'name_attr': self.name_attr_combo.currentText(),
            'background_value': self.background_value.value(),
            'ignore_background': self.ignore_background.isChecked(),
            'n_estimators': self.n_estimators.value(),
            'block_size': self.block_size.value(),
            'enable_sampling': self.enable_sampling.isChecked(),
            'max_samples': self.max_samples.value(),
            'fast_mode': self.fast_mode.isChecked(),
            'enable_hyperparameter_optimization': self.enable_hyperparameter_optimization.isChecked(),
            'hyperparameter_optimization_method': self.hyperparameter_optimization_method.currentText(),
            'hyperparameter_iterations': self.hyperparameter_iterations.value(),
            'enable_postprocessing': self.enable_postprocessing.isChecked(),
            'postprocessing_method': self.postprocessing_method.currentText(),
            'postprocessing_size': self.postprocessing_size.value(),
            'enable_feature_importance': self.enable_feature_importance.isChecked(),
            'enable_roc_analysis': self.enable_roc_analysis.isChecked(),
            'enable_model_saving': self.enable_model_saving.isChecked(),
            'selected_classifiers': {code: checkbox.isChecked() for code, checkbox in self.classifier_vars.items()},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename, _ = QFileDialog.getSaveFileName(self, "保存配置", "", "JSON文件 (*.json)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "成功", f"配置已保存到: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
    
    def load_config(self):
        """加载配置"""
        filename, _ = QFileDialog.getOpenFileName(self, "加载配置", "", "JSON文件 (*.json)")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 恢复配置
                self.image_selector.set_path(config.get('image_path', ''))
                self.train_shp_selector.set_path(config.get('train_shp_path', ''))
                self.val_shp_selector.set_path(config.get('val_shp_path', ''))
                self.output_selector.set_path(config.get('output_dir', './results_gui'))
                self.class_attr_combo.setCurrentText(config.get('class_attr', ''))
                self.name_attr_combo.setCurrentText(config.get('name_attr', ''))
                self.background_value.setValue(config.get('background_value', 0))
                self.ignore_background.setChecked(config.get('ignore_background', True))
                self.n_estimators.setValue(config.get('n_estimators', 100))
                self.block_size.setValue(config.get('block_size', 512))
                self.enable_sampling.setChecked(config.get('enable_sampling', True))
                self.max_samples.setValue(config.get('max_samples', 50000))
                self.fast_mode.setChecked(config.get('fast_mode', False))
                
                # 新功能配置
                self.enable_hyperparameter_optimization.setChecked(config.get('enable_hyperparameter_optimization', False))
                self.hyperparameter_optimization_method.setCurrentText(config.get('hyperparameter_optimization_method', 'random'))
                self.hyperparameter_iterations.setValue(config.get('hyperparameter_iterations', 20))
                self.enable_postprocessing.setChecked(config.get('enable_postprocessing', True))
                self.postprocessing_method.setCurrentText(config.get('postprocessing_method', 'majority'))
                self.postprocessing_size.setValue(config.get('postprocessing_size', 3))
                self.enable_feature_importance.setChecked(config.get('enable_feature_importance', True))
                self.enable_roc_analysis.setChecked(config.get('enable_roc_analysis', True))
                self.enable_model_saving.setChecked(config.get('enable_model_saving', True))
                
                # 分类器选择
                selected_classifiers = config.get('selected_classifiers', {})
                for code, checkbox in self.classifier_vars.items():
                    checkbox.setChecked(selected_classifiers.get(code, False))
                
                QMessageBox.information(self, "成功", f"配置已从 {filename} 加载")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
    
    def log(self, message):
        """添加日志"""
        self.log_widget.append(message)
    
    def update_accuracy_plot(self, comparison_results):
        """更新精度对比图"""
        if not comparison_results:
            return
        
        self.accuracy_widget.clear()
        ax = self.accuracy_widget.axes
        
        df = pd.DataFrame(comparison_results)
        
        # 创建子图
        ax1 = self.accuracy_widget.figure.add_subplot(121)
        ax2 = self.accuracy_widget.figure.add_subplot(122)
        
        # 精度对比
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x - width/2, df['训练集精度'], width, label='训练集', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, df['验证集精度'], width, label='验证集', alpha=0.8, color='coral')
        
        ax1.set_xlabel('分类器', fontsize=11)
        ax1.set_ylabel('精度', fontsize=11)
        ax1.set_title('总体精度对比', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['分类器名称'], rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.05])
        
        # 添加数值标签
        for i, (train_acc, val_acc) in enumerate(zip(df['训练集精度'], df['验证集精度'])):
            ax1.text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
            ax1.text(i + width/2, val_acc + 0.01, f'{val_acc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Kappa对比
        ax2.bar(x - width/2, df['训练集Kappa'], width, label='训练集', alpha=0.8, color='steelblue')
        ax2.bar(x + width/2, df['验证集Kappa'], width, label='验证集', alpha=0.8, color='coral')
        
        ax2.set_xlabel('分类器', fontsize=11)
        ax2.set_ylabel('Kappa系数', fontsize=11)
        ax2.set_title('Kappa系数对比', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['分类器名称'], rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.05])
        
        self.accuracy_widget.figure.tight_layout()
        self.accuracy_widget.canvas.draw()
    
    def update_confusion_matrix(self, y_true, y_pred, class_names):
        """更新混淆矩阵显示"""
        self.cm_widget.clear()
        ax = self.cm_widget.axes
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': '样本数量'}, ax=ax)
        
        ax.set_xlabel('预测类别', fontsize=11)
        ax.set_ylabel('真实类别', fontsize=11)
        ax.set_title('最佳分类器混淆矩阵', fontsize=12, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        self.cm_widget.figure.tight_layout()
        self.cm_widget.canvas.draw()
    
    def update_time_plot(self, comparison_results):
        """更新时间对比图"""
        if not comparison_results:
            return
        
        self.time_widget.clear()
        ax = self.time_widget.axes
        
        df = pd.DataFrame(comparison_results)
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['训练时间(秒)'], width, label='训练时间', 
                      alpha=0.8, color='lightgreen')
        bars2 = ax.bar(x + width/2, df['预测时间(秒)'], width, label='预测时间', 
                      alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('分类器', fontsize=11)
        ax.set_ylabel('时间 (秒)', fontsize=11)
        ax.set_title('训练和预测时间对比', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['分类器名称'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        self.time_widget.figure.tight_layout()
        self.time_widget.canvas.draw()
    
    def update_feature_importance_plot(self, feature_importance_data, band_names):
        """更新特征重要性图"""
        if not feature_importance_data:
            # 绘制空图表
            self.feature_widget.clear()
            ax = self.feature_widget.axes
            ax.text(0.5, 0.5, '无特征重要性数据', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('特征重要性分析')
            ax.axis('off')
            self.feature_widget.canvas.draw()
            return
        
        self.feature_widget.clear()
        ax = self.feature_widget.axes
        
        # 只显示最佳分类器的特征重要性
        best_clf_code = None
        if self.comparison_results:
            df = pd.DataFrame(self.comparison_results)
            best_idx = df['验证集精度'].idxmax()
            best_clf_code = df.loc[best_idx, '分类器代码']
            clf_name = df.loc[best_idx, '分类器名称']
        
        if best_clf_code and best_clf_code in feature_importance_data:
            importance_data = feature_importance_data[best_clf_code]
            
            indices = importance_data['indices']
            importances = importance_data['importances']
            
            if len(importances) == 0:
                ax.text(0.5, 0.5, '特征重要性数据为空', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{clf_name} - 特征重要性分析')
                ax.axis('off')
            else:
                # 确保索引不超出范围
                valid_indices = indices[indices < len(importances)]
                if len(valid_indices) == 0:
                    valid_indices = np.arange(len(importances))
                
                sorted_importances = importances[valid_indices]
                
                # 确保有足够的特征名称
                if len(band_names) < len(sorted_importances):
                    # 补充特征名称
                    extra_features = len(sorted_importances) - len(band_names)
                    band_names.extend([f"特征_{i+1}" for i in range(len(band_names), len(band_names) + extra_features)])
                
                features = [band_names[i] for i in valid_indices if i < len(band_names)]
                
                y_pos = np.arange(len(features))
                
                # 处理标准差
                if 'std' in importance_data and importance_data['std'] is not None:
                    std = importance_data['std']
                    if len(std) == len(importances):
                        sorted_std = std[valid_indices]
                        ax.barh(y_pos, sorted_importances, xerr=sorted_std, 
                            align='center', alpha=0.7, color='steelblue')
                    else:
                        ax.barh(y_pos, sorted_importances, align='center', 
                            alpha=0.7, color='steelblue')
                else:
                    ax.barh(y_pos, sorted_importances, align='center', 
                        alpha=0.7, color='steelblue')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('特征重要性')
                ax.set_title(f'{clf_name} - 特征重要性分析', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
        
        self.feature_widget.figure.tight_layout()
        self.feature_widget.canvas.draw()
    
    def update_roc_plot(self, roc_data_dict):
        """更新ROC曲线图"""
        if not roc_data_dict:
            # 绘制空的ROC曲线
            self.roc_widget.clear()
            ax = self.roc_widget.axes
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('假正率')
            ax.set_ylabel('真正率')
            ax.set_title('ROC曲线')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            self.roc_widget.figure.tight_layout()
            self.roc_widget.canvas.draw()
            return
        
        # 只显示最佳分类器的ROC曲线
        best_clf_code = None
        if self.comparison_results:
            df = pd.DataFrame(self.comparison_results)
            best_idx = df['验证集精度'].idxmax()
            best_clf_code = df.loc[best_idx, '分类器代码']
            clf_name = df.loc[best_idx, '分类器名称']
        
        if best_clf_code and best_clf_code in roc_data_dict:
            roc_data = roc_data_dict[best_clf_code]
            
            self.roc_widget.clear()
            ax = self.roc_widget.axes
            
            # 从保存的数据中绘制ROC曲线
            fpr = roc_data.get('fpr', {})
            tpr = roc_data.get('tpr', {})
            auc_scores = roc_data.get('auc', {})
            
            if isinstance(fpr, dict) and isinstance(tpr, dict):
                # 多分类情况
                colors = plt.cm.Set1(np.linspace(0, 1, len(fpr)))
                
                for i, (key, fpr_val) in enumerate(fpr.items()):
                    if key == 'macro':
                        # 宏平均使用特殊样式
                        ax.plot(fpr_val, tpr[key], color='navy', linestyle=':', linewidth=4,
                            label=f'宏平均 (AUC = {auc_scores.get(key, 0):.2f})')
                    elif key in auc_scores:
                        # 单个类别
                        class_name = f'类别_{key}' if isinstance(key, int) else str(key)
                        ax.plot(fpr_val, tpr[key], color=colors[i % len(colors)], lw=2,
                            label=f'{class_name} (AUC = {auc_scores[key]:.2f})')
                
                # 添加随机分类器线
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                    label='随机分类器', alpha=0.8)
                
            else:
                # 二分类情况
                ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC曲线 (AUC = {auc_scores:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='随机分类器')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
            ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
            ax.set_title(f'{clf_name} - ROC曲线', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            self.roc_widget.figure.tight_layout()
            self.roc_widget.canvas.draw()
    
    def update_result_preview(self, image_path, classified_path, class_names, class_colors):
        """更新分类结果预览"""
        try:
            self.result_widget.clear()
            
            # 读取影像和分类结果
            img = rxr.open_rasterio(image_path, masked=True)
            classified = rxr.open_rasterio(classified_path)
            
            # 创建子图
            ax1 = self.result_widget.figure.add_subplot(121)
            ax2 = self.result_widget.figure.add_subplot(122)
            
            # 显示原始影像
            if img.shape[0] >= 3:
                rgb_data = np.moveaxis(img.values[:3], 0, -1)
                # 处理可能的NaN值
                rgb_data = np.nan_to_num(rgb_data, nan=0.0)
                p2, p98 = np.percentile(rgb_data[rgb_data > 0], (2, 98))
                rgb_display = np.clip((rgb_data - p2) / (p98 - p2), 0, 1)
                ax1.imshow(rgb_display)
            else:
                single_band = np.nan_to_num(img.values[0], nan=0.0)
                ax1.imshow(single_band, cmap='gray')
            
            ax1.set_title('原始遥感影像', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # 显示分类结果
            classified_data = classified.values.squeeze()
            
            # 获取类别
            classes = np.unique(classified_data)
            classes = classes[classes > 0]  # 排除背景值
            
            if len(classes) > 0:
                # 创建颜色映射
                colors = [class_colors.get(c, 'black') for c in classes]
                labels = [class_names.get(c, f'类别_{c}') for c in classes]
                
                cmap = mcolors.ListedColormap(colors)
                bounds = np.append(classes, classes[-1] + 1) - 0.5
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                
                # 背景设为透明
                display_data = classified_data.astype(float)
                display_data[classified_data == 0] = np.nan
                
                im = ax2.imshow(display_data, cmap=cmap, norm=norm)
                ax2.set_title('分类结果（最佳分类器）', fontsize=12, fontweight='bold')
                ax2.axis('off')
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=color, label=label) 
                                  for color, label in zip(colors, labels)]
                ax2.legend(handles=legend_elements, loc='upper left', 
                          bbox_to_anchor=(1.05, 1), fontsize=9)
            else:
                ax2.text(0.5, 0.5, '无分类结果', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title('分类结果（最佳分类器）', fontsize=12, fontweight='bold')
                ax2.axis('off')
            
            self.result_widget.figure.tight_layout()
            self.result_widget.canvas.draw()
            
        except Exception as e:
            self.log(f"预览显示错误: {str(e)}")
    
    def start_classification(self):
        """开始分类"""
        # 检查输入
        if not self.image_selector.get_path():
            QMessageBox.critical(self, "错误", "请选择影像文件！")
            return
        
        if not self.train_shp_selector.get_path():
            QMessageBox.critical(self, "错误", "请选择训练样本！")
            return
        
        if not self.class_attr_combo.currentText():
            QMessageBox.critical(self, "错误", "请选择类别编号字段！")
            return
        
        selected_classifiers = [code for code, checkbox in self.classifier_vars.items() if checkbox.isChecked()]
        if not selected_classifiers:
            QMessageBox.critical(self, "错误", "请至少选择一个分类器！")
            return
        
        # 性能警告
        all_classifiers = self.backend.get_all_classifiers()
        very_slow_clfs = []
        
        for code in selected_classifiers:
            if code in all_classifiers:
                speed_tag = all_classifiers[code][5]
                name = all_classifiers[code][1]
                if speed_tag == "very_slow":
                    very_slow_clfs.append(name)
        
        if very_slow_clfs:
            warning_msg = "⚠️ 以下分类器预测非常慢:\n"
            for clf in very_slow_clfs:
                warning_msg += f"  • {clf}\n"
            warning_msg += "\n是否继续?"
            
            reply = QMessageBox.question(self, "性能警告", warning_msg, 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.is_running = True
        
        # 清空日志
        self.log_widget.clear()
        
        self.log("="*80)
        self.log("  遥感影像监督分类系统 v5.1 - 专业版 (PyQt5)")
        self.log("="*80)
        self.log(f"选择的分类器: {len(selected_classifiers)} 个")
        self.log(f"背景值: {self.background_value.value()}")
        self.log("")
        
        # 切换到日志标签页
        self.tab_widget.setCurrentIndex(0)
        
        # 创建配置字典
        config = {
            'image_path': self.image_selector.get_path(),
            'train_shp_path': self.train_shp_selector.get_path(),
            'val_shp_path': self.val_shp_selector.get_path(),
            'output_dir': self.output_selector.get_path(),
            'class_attr': self.class_attr_combo.currentText(),
            'name_attr': self.name_attr_combo.currentText(),
            'background_value': self.background_value.value(),
            'ignore_background': self.ignore_background.isChecked(),
            'n_estimators': self.n_estimators.value(),
            'block_size': self.block_size.value(),
            'enable_sampling': self.enable_sampling.isChecked(),
            'max_samples': self.max_samples.value(),
            'fast_mode': self.fast_mode.isChecked(),
            'enable_hyperparameter_optimization': self.enable_hyperparameter_optimization.isChecked(),
            'hyperparameter_optimization_method': self.hyperparameter_optimization_method.currentText(),
            'hyperparameter_iterations': self.hyperparameter_iterations.value(),
            'enable_postprocessing': self.enable_postprocessing.isChecked(),
            'postprocessing_method': self.postprocessing_method.currentText(),
            'postprocessing_size': self.postprocessing_size.value(),
            'enable_feature_importance': self.enable_feature_importance.isChecked(),
            'enable_roc_analysis': self.enable_roc_analysis.isChecked(),
            'enable_model_saving': self.enable_model_saving.isChecked(),
        }
        
        # 启动分类线程
        self.classification_thread = ClassificationThread(self.backend, config, selected_classifiers)
        self.classification_thread.progress_signal.connect(self.progress_bar.setValue)
        self.classification_thread.log_signal.connect(self.log)
        self.classification_thread.status_signal.connect(self.status_label.setText)
        self.classification_thread.finished_signal.connect(self.classification_finished)
        self.classification_thread.start()
    
    def stop_classification(self):
        """停止分类"""
        self.is_running = False
        if self.classification_thread:
            self.classification_thread.stop()
        self.log("\n⏸ 用户请求停止...")
        self.status_label.setText("已停止")
    
    def try_generate_result_preview(self):
        """尝试生成结果预览"""
        if not self.comparison_results or not self.best_result_path:
            return
        
        try:
            # 获取最佳分类器结果
            df = pd.DataFrame(self.comparison_results)
            best_result = df.loc[df['验证集精度'].idxmax()]
            best_clf_code = best_result['分类器代码']
            
            # 构建结果路径
            out_dir = Path(self.output_selector.get_path())
            clf_dir = out_dir / best_clf_code
            classified_path = clf_dir / f"classified_{best_clf_code}.tif"
            
            if not classified_path.exists():
                return
            
            # 获取类别信息
            class_names, class_colors, _ = self.backend.get_class_info_from_shp(
                self.train_shp_selector.get_path(),
                self.class_attr_combo.currentText(),
                self.name_attr_combo.currentText()
            )
            
            # 更新预览
            self.update_result_preview(
                self.image_selector.get_path(),
                str(classified_path),
                class_names,
                class_colors
            )
            
        except Exception as e:
            self.log(f"生成结果预览失败: {str(e)}")

    def classification_finished(self, success, message, comparison_results, confusion_data, 
                           feature_importance_data, roc_data_dict, result_preview_data):
        """分类完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        if success:
            self.status_label.setText("✅ 完成!")
            self.comparison_results = comparison_results
            self.feature_importance_data = feature_importance_data
            self.roc_data_dict = roc_data_dict
            
            # 更新所有图表
            if comparison_results:
                self.update_accuracy_plot(comparison_results)
                self.update_time_plot(comparison_results)
            
            if confusion_data:
                y_true, y_pred, class_names = confusion_data
                self.update_confusion_matrix(y_true, y_pred, class_names)
            
            if feature_importance_data:
                # 修复：从分类线程获取真实的波段名称
                try:
                    # 重新读取影像获取波段信息
                    img = rxr.open_rasterio(self.image_selector.get_path(), masked=True)
                    band_names = []
                    if hasattr(img, 'long_name') and img.long_name:
                        band_names = img.long_name
                    elif hasattr(img, 'band') and img.band.values is not None:
                        band_names = [f"波段_{b}" for b in img.band.values]
                    else:
                        band_names = [f"波段_{i+1}" for i in range(img.shape[0])]
                except:
                    band_names = [f"波段_{i+1}" for i in range(10)]  # 备用方案
                
                self.update_feature_importance_plot(feature_importance_data, band_names)
            
            if roc_data_dict:
                self.update_roc_plot(roc_data_dict)
            
            # 修复：添加结果预览更新
            if result_preview_data:
                image_path, classified_path, class_names, class_colors = result_preview_data
                self.update_result_preview(image_path, classified_path, class_names, class_colors)
            else:
                # 如果没有预览数据，尝试从最佳结果生成
                self.try_generate_result_preview()
            
            QMessageBox.information(self, "完成", message)
        else:
            self.status_label.setText("❌ 错误")
            if "错误" in message:
                QMessageBox.critical(self, "错误", message)
            else:
                QMessageBox.warning(self, "警告", message)
    
    def open_result_dir(self):
        """打开结果目录"""
        out_dir = Path(self.output_selector.get_path())
        if out_dir.exists():
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(out_dir)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])
        else:
            QMessageBox.warning(self, "警告", "结果目录不存在！")


# ==================== 主程序入口 ====================

def main():
    """程序入口"""
    print("="*80)
    print("  遥感影像监督分类系统 v1.0 - 专业版 (PyQt5)")
    print("="*80)
    print("\n正在检查依赖库...")
    
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = ClassificationGUI()
    window.show()
    
    # 显示欢迎信息
    window.log("="*80)
    window.log("  遥感影像监督分类系统 v1.0 - 专业版 (PyQt5)")
    window.log("                     @ 3S&ML实验室")
    window.log("="*80)
    window.log("\n核心功能:")
    window.log("  ✅ 多分类器支持: 集成12+种机器学习分类器")
    window.log("  ✅ 超参数自动优化: 自动寻找最佳参数组合")
    window.log("  ✅ 后处理滤波: 提升分类结果质量")
    window.log("  ✅ ROC曲线分析: 专业模型评估")
    window.log("  ✅ 特征重要性分析: 波段贡献度分析")
    window.log("  ✅ 模型保存/加载: 支持模型复用")
    window.log("  ✅ 批量处理: 多分类器并行对比")
    window.log("  ✅ 可视化分析: 实时图表显示")
    window.log("\n使用流程:")
    window.log("  1. 选择影像和样本文件")
    window.log("  2. 点击'刷新字段列表'选择类别字段")
    window.log("  3. 设置背景值和其他参数")
    window.log("  4. 选择分类器")
    window.log("  5. 配置高级功能（超参数优化、后处理等）")
    window.log("  6. 点击'开始分类'")
    window.log("  7. 查看右侧实时图表和各分类器详细结果")
    window.log("="*80)
    window.log("\n技术优势:")
    window.log("  🚀 高性能: 支持分块处理大影像")
    window.log("  📊 专业评估: 提供完整的精度评价指标")
    window.log("  💾 多种格式: 支持多种数据格式输出")
    window.log("  🔧 灵活配置: 参数可调，适应不同需求")
    window.log("")
    
    print("\n✓ 系统启动成功!")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遥感影像监督分类系统 - 专业版 v4.1
=====================================
新增:
- 完善结果预览显示
- Excel格式报告输出
- 混淆矩阵可视化
- 图表实时刷新优化
"""

import os
import sys
import time
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from rasterio import features
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
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
                           cohen_kappa_score, precision_score, recall_score, f1_score)
from scipy import ndimage
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

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
            
            "ada": (AdaBoostClassifier(n_estimators=n_est, learning_rate=1.0, 
                                      random_state=self.RANDOM_STATE, algorithm='SAMME.R'),
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
    
    def estimate_prediction_time(self, clf_code, n_pixels, speed_tag):
        """估算预测时间"""
        time_per_million = {"very_fast": 1, "fast": 3, "medium": 10, "slow": 30, "very_slow": 300}
        base_time = time_per_million.get(speed_tag, 10)
        return (n_pixels / 1_000_000) * base_time
    
    def predict_by_block(self, model, image, out_path, block_size=512, 
                        ignore_background=True, background_value=0, progress_callback=None,
                        label_encoder=None, scaler=None):
        """分块预测"""
        height, width = image.shape[1], image.shape[2]
        prediction = np.zeros((height, width), dtype='uint16')
        
        if ignore_background:
            background_mask = self.get_background_mask(image, background_value)
        
        total_blocks = int(np.ceil(height / block_size))
        
        for i, y in enumerate(range(0, height, block_size)):
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
        
        # 保存结果
        prediction_da = xr.DataArray(prediction, dims=['y', 'x'],
                                     coords={'y': image.coords['y'], 'x': image.coords['x']})
        
        prediction_da.rio.write_crs(image.rio.crs, inplace=True)
        prediction_da.rio.write_transform(image.rio.transform(), inplace=True)
        prediction_da.rio.write_nodata(background_value, inplace=True)
        
        prediction_da.rio.to_raster(out_path, driver='GTiff', dtype='uint16', 
                                    compress='lzw', tiled=True)
        return out_path

# ==================== GUI主类 ====================
class ClassificationGUI:
    """遥感影像分类GUI主界面（专业版）"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("遥感影像监督分类系统 v4.1 - 专业版")
        self.root.geometry("1600x900")
        
        # 后端处理对象
        self.backend = ClassificationBackend()
        
        # 数据变量
        self.image_path = tk.StringVar()
        self.train_shp_path = tk.StringVar()
        self.val_shp_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path("./results_gui")))
        
        # 字段选择
        self.train_fields = []
        self.class_attr = tk.StringVar()
        self.name_attr = tk.StringVar()
        
        # 背景值
        self.background_value = tk.IntVar(value=0)
        self.ignore_background = tk.BooleanVar(value=True)
        
        # 其他参数
        self.n_estimators = tk.IntVar(value=100)
        self.block_size = tk.IntVar(value=512)
        
        # 性能优化参数
        self.enable_sampling = tk.BooleanVar(value=True)
        self.max_samples = tk.IntVar(value=50000)
        self.fast_mode = tk.BooleanVar(value=False)
        
        # 分类器选择
        self.classifier_vars = {}
        all_classifiers = self.backend.get_all_classifiers()
        for code in all_classifiers.keys():
            self.classifier_vars[code] = tk.BooleanVar(value=False)
        
        # 运行状态
        self.is_running = False
        self.log_queue = queue.Queue()
        
        # 存储结果数据
        self.comparison_results = []
        self.current_confusion_matrix = None
        self.current_y_true = None
        self.current_y_pred = None
        self.class_names_dict = {}
        self.class_colors_dict = {}
        self.best_result_path = None
        
        # 构建界面
        self.build_ui()
        
        # 启动日志更新
        self.update_log()
    
    def build_ui(self):
        """构建用户界面（左右分栏）"""
        # 创建主PanedWindow
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== 左侧面板：参数设置 =====
        left_frame = ttk.Frame(main_paned, width=600)
        main_paned.add(left_frame, weight=1)
        
        # 创建滚动区域
        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_left = ttk.Frame(canvas)
        
        scrollable_left.bind("<Configure>", 
                            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_left, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 1. 文件选择
        file_frame = ttk.LabelFrame(scrollable_left, text="📁 数据文件", padding="10")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="影像文件:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Entry(file_frame, textvariable=self.image_path, width=40).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(file_frame, text="浏览", command=self.browse_image).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="训练样本:").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Entry(file_frame, textvariable=self.train_shp_path, width=40).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(file_frame, text="浏览", command=self.browse_train_shp).grid(row=1, column=2)
        
        ttk.Label(file_frame, text="验证样本:").grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Entry(file_frame, textvariable=self.val_shp_path, width=40).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(file_frame, text="浏览", command=self.browse_val_shp).grid(row=2, column=2)
        
        ttk.Label(file_frame, text="输出目录:").grid(row=3, column=0, sticky=tk.W, pady=3)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=40).grid(
            row=3, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(file_frame, text="浏览", command=self.browse_output).grid(row=3, column=2)
        
        file_frame.columnconfigure(1, weight=1)
        
        # 2. 字段选择
        field_frame = ttk.LabelFrame(scrollable_left, text="🏷️ 字段配置", padding="10")
        field_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(field_frame, text="类别编号字段:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.class_attr_combo = ttk.Combobox(field_frame, textvariable=self.class_attr, 
                                            width=20, state="readonly")
        self.class_attr_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(field_frame, text="类别名称字段:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.name_attr_combo = ttk.Combobox(field_frame, textvariable=self.name_attr, 
                                           width=20, state="readonly")
        self.name_attr_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(field_frame, text="🔄 刷新字段列表", 
                  command=self.refresh_fields).grid(row=0, column=2, rowspan=2, padx=5)
        
        field_frame.columnconfigure(1, weight=1)
        
        # 3. 背景值设置
        bg_frame = ttk.LabelFrame(scrollable_left, text="🎨 背景值设置", padding="10")
        bg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(bg_frame, text="忽略背景值", 
                       variable=self.ignore_background).grid(row=0, column=0, sticky=tk.W, pady=3)
        
        ttk.Label(bg_frame, text="背景值:").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(bg_frame, from_=-9999, to=9999, textvariable=self.background_value, 
                   width=15).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(bg_frame, text="(默认0, 常见: -9999, 255)", 
                 font=('', 8), foreground='gray').grid(row=1, column=2, sticky=tk.W)
        
        # 4. 分类参数
        param_frame = ttk.LabelFrame(scrollable_left, text="⚙️ 分类参数", padding="10")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="树模型数量:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(param_frame, from_=10, to=500, textvariable=self.n_estimators, 
                   width=15).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(param_frame, text="分块大小:").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(param_frame, from_=256, to=2048, increment=256, 
                   textvariable=self.block_size, width=15).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # 5. 性能优化
        opt_frame = ttk.LabelFrame(scrollable_left, text="⚡ 性能优化", padding="10")
        opt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        sample_frame = ttk.Frame(opt_frame)
        sample_frame.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(sample_frame, text="启用采样", 
                       variable=self.enable_sampling,
                       command=self.toggle_sampling).pack(side=tk.LEFT)
        
        ttk.Label(sample_frame, text="最大样本数:").pack(side=tk.LEFT, padx=(10, 0))
        self.max_samples_spinbox = ttk.Spinbox(sample_frame, from_=10000, to=200000, 
                                              increment=10000, textvariable=self.max_samples, 
                                              width=10)
        self.max_samples_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(opt_frame, text="快速模式", 
                       variable=self.fast_mode).pack(anchor=tk.W, pady=2)
        
        # 6. 分类器选择
        clf_frame = ttk.LabelFrame(scrollable_left, text="🤖 分类器选择", padding="10")
        clf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 快捷按钮
        btn_frame = ttk.Frame(clf_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(btn_frame, text="全选", command=self.select_all_classifiers, 
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="全不选", command=self.deselect_all_classifiers, 
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="✓推荐", command=self.select_recommended, 
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="⚡快速", command=self.select_fast, 
                  width=10).pack(side=tk.LEFT, padx=2)
        
        # 分类器复选框
        all_classifiers = self.backend.get_all_classifiers()
        
        clf_canvas = tk.Canvas(clf_frame, height=150)
        clf_scrollbar = ttk.Scrollbar(clf_frame, orient="vertical", command=clf_canvas.yview)
        clf_scrollable = ttk.Frame(clf_canvas)
        
        clf_scrollable.bind("<Configure>", 
                           lambda e: clf_canvas.configure(scrollregion=clf_canvas.bbox("all")))
        
        clf_canvas.create_window((0, 0), window=clf_scrollable, anchor="nw")
        clf_canvas.configure(yscrollcommand=clf_scrollbar.set)
        
        # SVM组
        ttk.Label(clf_scrollable, text="SVM:", font=('', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        row = 1
        svm_codes = ["svm_linear", "linear_svc", "sgd_svm", "nystroem_svm", 
                     "rbf_sampler_svm", "svm_rbf"]
        for code in svm_codes:
            if code in all_classifiers:
                _, name, _, _, _, _ = all_classifiers[code]
                ttk.Checkbutton(clf_scrollable, text=name, 
                              variable=self.classifier_vars[code]).grid(
                    row=row, column=0, sticky=tk.W, padx=20
                )
                row += 1
        
        # 树模型
        ttk.Label(clf_scrollable, text="树模型:", font=('', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(5, 2)
        )
        row += 1
        tree_codes = ["rf", "et", "dt", "xgb", "lgb", "gb", "ada"]
        for code in tree_codes:
            if code in all_classifiers:
                _, name, _, _, _, _ = all_classifiers[code]
                ttk.Checkbutton(clf_scrollable, text=name,
                              variable=self.classifier_vars[code]).grid(
                    row=row, column=0, sticky=tk.W, padx=20
                )
                row += 1
        
        # 其他
        ttk.Label(clf_scrollable, text="其他:", font=('', 9, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(5, 2)
        )
        row += 1
        other_codes = ["knn", "nb", "lr", "mlp"]
        for code in other_codes:
            if code in all_classifiers:
                _, name, _, _, _, _ = all_classifiers[code]
                ttk.Checkbutton(clf_scrollable, text=name,
                              variable=self.classifier_vars[code]).grid(
                    row=row, column=0, sticky=tk.W, padx=20
                )
                row += 1
        
        clf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        clf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 7. 控制按钮
        control_frame = ttk.LabelFrame(scrollable_left, text="🎮 运行控制", padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_control_frame = ttk.Frame(control_frame)
        btn_control_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(btn_control_frame, text="▶ 开始分类", 
                                    command=self.start_classification, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_control_frame, text="⏸ 停止", 
                                   command=self.stop_classification, 
                                   state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_control_frame, text="📁 打开结果", 
                  command=self.open_result_dir, width=15).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        ttk.Label(control_frame, text="进度:").pack(anchor=tk.W, pady=(10, 0))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 状态
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(control_frame, textvariable=self.status_var, 
                 relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)
        
        # ===== 右侧面板：图件显示 =====
        right_frame = ttk.Frame(main_paned, width=900)
        main_paned.add(right_frame, weight=2)
        
        # 创建Notebook
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 标签页1：运行日志
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="📝 运行日志")
        
        self.log_text = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, 
                                                  font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 标签页2：精度对比
        accuracy_tab = ttk.Frame(self.notebook)
        self.notebook.add(accuracy_tab, text="📊 精度对比")
        
        self.accuracy_fig = Figure(figsize=(10, 6), dpi=100)
        self.accuracy_canvas = FigureCanvasTkAgg(self.accuracy_fig, accuracy_tab)
        self.accuracy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_acc = ttk.Frame(accuracy_tab)
        toolbar_acc.pack(fill=tk.X)
        NavigationToolbar2Tk(self.accuracy_canvas, toolbar_acc)
        
        # 标签页3：混淆矩阵
        cm_tab = ttk.Frame(self.notebook)
        self.notebook.add(cm_tab, text="🔥 混淆矩阵")
        
        self.cm_fig = Figure(figsize=(8, 6), dpi=100)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, cm_tab)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_cm = ttk.Frame(cm_tab)
        toolbar_cm.pack(fill=tk.X)
        NavigationToolbar2Tk(self.cm_canvas, toolbar_cm)
        
        # 标签页4：时间对比
        time_tab = ttk.Frame(self.notebook)
        self.notebook.add(time_tab, text="⏱️ 时间对比")
        
        self.time_fig = Figure(figsize=(10, 6), dpi=100)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, time_tab)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_time = ttk.Frame(time_tab)
        toolbar_time.pack(fill=tk.X)
        NavigationToolbar2Tk(self.time_canvas, toolbar_time)
        
        # 标签页5：分类结果预览
        result_tab = ttk.Frame(self.notebook)
        self.notebook.add(result_tab, text="🗺️ 结果预览")
        
        self.result_fig = Figure(figsize=(10, 6), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, result_tab)
        self.result_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_result = ttk.Frame(result_tab)
        toolbar_result.pack(fill=tk.X)
        NavigationToolbar2Tk(self.result_canvas, toolbar_result)
    
    # ===== 辅助函数 =====
    def toggle_sampling(self):
        if self.enable_sampling.get():
            self.max_samples_spinbox.config(state=tk.NORMAL)
        else:
            self.max_samples_spinbox.config(state=tk.DISABLED)
    
    def refresh_fields(self):
        train_shp = self.train_shp_path.get()
        if not train_shp or not os.path.exists(train_shp):
            messagebox.showwarning("警告", "请先选择训练样本文件！")
            return
        
        fields = self.backend.get_shapefile_fields(train_shp)
        if fields:
            fields = [f for f in fields if f.lower() != 'geometry']
            
            self.class_attr_combo['values'] = fields
            self.name_attr_combo['values'] = fields
            
            if 'class' in fields:
                self.class_attr.set('class')
            elif 'Class' in fields:
                self.class_attr.set('Class')
            elif fields:
                self.class_attr.set(fields[0])
            
            if 'name' in fields:
                self.name_attr.set('name')
            elif 'Name' in fields:
                self.name_attr.set('Name')
            elif len(fields) > 1:
                self.name_attr.set(fields[1])
            elif fields:
                self.name_attr.set(fields[0])
            
            messagebox.showinfo("成功", f"已加载 {len(fields)} 个字段")
        else:
            messagebox.showerror("错误", "无法读取字段列表！")
    
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="选择影像文件",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("所有文件", "*.*")]
        )
        if filename:
            self.image_path.set(filename)
            self.status_var.set(f"已选择影像: {Path(filename).name}")
    
    def browse_train_shp(self):
        filename = filedialog.askopenfilename(
            title="选择训练样本",
            filetypes=[("Shapefile", "*.shp"), ("所有文件", "*.*")]
        )
        if filename:
            self.train_shp_path.set(filename)
            self.refresh_fields()
    
    def browse_val_shp(self):
        filename = filedialog.askopenfilename(
            title="选择验证样本",
            filetypes=[("Shapefile", "*.shp"), ("所有文件", "*.*")]
        )
        if filename:
            self.val_shp_path.set(filename)
    
    def browse_output(self):
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir.set(dirname)
    
    def select_all_classifiers(self):
        for var in self.classifier_vars.values():
            var.set(True)
    
    def deselect_all_classifiers(self):
        for var in self.classifier_vars.values():
            var.set(False)
    
    def select_recommended(self):
        recommended = ["rf", "xgb", "et", "lgb", "linear_svc", "nystroem_svm"]
        for code, var in self.classifier_vars.items():
            var.set(code in recommended)
    
    def select_fast(self):
        fast = ["rf", "et", "dt", "xgb", "lgb", "nb", "lr", "sgd_svm", "linear_svc"]
        for code, var in self.classifier_vars.items():
            var.set(code in fast)
    
    def log(self, message):
        self.log_queue.put(message)
    
    def update_log(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.update_log)
    
    def update_accuracy_plot(self):
        """更新精度对比图"""
        if not self.comparison_results:
            return
        
        df = pd.DataFrame(self.comparison_results)
        
        self.accuracy_fig.clear()
        
        # 创建子图
        ax1 = self.accuracy_fig.add_subplot(121)
        ax2 = self.accuracy_fig.add_subplot(122)
        
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
        
        self.accuracy_fig.tight_layout()
        self.accuracy_canvas.draw()
    
    def update_confusion_matrix(self, y_true, y_pred, class_names):
        """更新混淆矩阵显示"""
        self.cm_fig.clear()
        ax = self.cm_fig.add_subplot(111)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': '样本数量'}, ax=ax)
        
        ax.set_xlabel('预测类别', fontsize=11)
        ax.set_ylabel('真实类别', fontsize=11)
        ax.set_title('最佳分类器混淆矩阵', fontsize=12, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        self.cm_fig.tight_layout()
        self.cm_canvas.draw()
    
    def update_time_plot(self):
        """更新时间对比图"""
        if not self.comparison_results:
            return
        
        df = pd.DataFrame(self.comparison_results)
        
        self.time_fig.clear()
        ax = self.time_fig.add_subplot(111)
        
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
        
        self.time_fig.tight_layout()
        self.time_canvas.draw()
    
    def update_result_preview(self, image_path, classified_path, class_names, class_colors):
        """更新分类结果预览"""
        try:
            self.result_fig.clear()
            
            # 读取影像和分类结果
            img = rxr.open_rasterio(image_path, masked=True)
            classified = rxr.open_rasterio(classified_path)
            
            # 创建子图
            ax1 = self.result_fig.add_subplot(121)
            ax2 = self.result_fig.add_subplot(122)
            
            # 显示原始影像
            if img.shape[0] >= 3:
                rgb_data = np.moveaxis(img.values[:3], 0, -1)
                p2, p98 = np.percentile(rgb_data[rgb_data > 0], (2, 98))
                rgb_display = np.clip((rgb_data - p2) / (p98 - p2), 0, 1)
                ax1.imshow(rgb_display)
            else:
                ax1.imshow(img.values[0], cmap='gray')
            
            ax1.set_title('原始遥感影像', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # 显示分类结果
            classified_data = classified.values.squeeze()
            
            # 获取类别
            classes = np.unique(classified_data)
            classes = classes[classes > 0]
            
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
            
            self.result_fig.tight_layout()
            self.result_canvas.draw()
            
        except Exception as e:
            self.log(f"预览显示错误: {str(e)}")
    
    def export_to_excel(self, out_dir):
        """导出结果到Excel"""
        if not HAS_OPENPYXL:
            self.log("⚠️  未安装openpyxl，无法导出Excel")
            return
        
        if not self.comparison_results:
            return
        
        try:
            df = pd.DataFrame(self.comparison_results)
            
            excel_path = out_dir / "classification_comparison.xlsx"
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 主结果表
                df.to_excel(writer, sheet_name='分类器对比', index=False)
                
                # 获取工作簿和工作表
                workbook = writer.book
                worksheet = writer.sheets['分类器对比']
                
                # 设置列宽
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # 添加统计摘要表
                summary_data = {
                    '指标': ['最高精度', '最高Kappa', '最快训练', '最快预测'],
                    '分类器': [
                        df.loc[df['验证集精度'].idxmax(), '分类器名称'],
                        df.loc[df['验证集Kappa'].idxmax(), '分类器名称'],
                        df.loc[df['训练时间(秒)'].idxmin(), '分类器名称'],
                        df.loc[df['预测时间(秒)'].idxmin(), '分类器名称']
                    ],
                    '数值': [
                        f"{df['验证集精度'].max():.4f}",
                        f"{df['验证集Kappa'].max():.4f}",
                        f"{df['训练时间(秒)'].min():.2f}秒",
                        f"{df['预测时间(秒)'].min():.2f}秒"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='性能摘要', index=False)
            
            self.log(f"✓ Excel报告已保存: {excel_path}")
            
        except Exception as e:
            self.log(f"Excel导出失败: {str(e)}")
    
    def start_classification(self):
        """开始分类"""
        # 检查输入
        if not self.image_path.get():
            messagebox.showerror("错误", "请选择影像文件！")
            return
        
        if not self.train_shp_path.get():
            messagebox.showerror("错误", "请选择训练样本！")
            return
        
        if not self.class_attr.get():
            messagebox.showerror("错误", "请选择类别编号字段！")
            return
        
        selected_classifiers = [code for code, var in self.classifier_vars.items() if var.get()]
        if not selected_classifiers:
            messagebox.showerror("错误", "请至少选择一个分类器！")
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
            
            if not messagebox.askyesno("性能警告", warning_msg, icon='warning'):
                return
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.is_running = True
        
        # 清空
        self.log_text.delete(1.0, tk.END)
        self.comparison_results = []
        
        self.log("="*80)
        self.log("  遥感影像监督分类系统 v4.1")
        self.log("="*80)
        self.log(f"选择的分类器: {len(selected_classifiers)} 个")
        self.log(f"背景值: {self.background_value.get()}")
        self.log("")
        
        # 切换到日志标签页
        self.notebook.select(0)
        
        thread = threading.Thread(target=self.run_classification, args=(selected_classifiers,))
        thread.daemon = True
        thread.start()
    
    def stop_classification(self):
        self.is_running = False
        self.log("\n⏸ 用户请求停止...")
        self.status_var.set("已停止")
    
    def run_classification(self, selected_classifiers):
        """执行分类（主流程）"""
        try:
            out_dir = Path(self.output_dir.get())
            out_dir.mkdir(exist_ok=True)
            
            # 读取影像
            self.log(f"📁 读取影像...")
            self.status_var.set("读取影像...")
            img = rxr.open_rasterio(self.image_path.get(), masked=True)
            n_pixels = img.shape[1] * img.shape[2]
            self.log(f"   尺寸: {img.shape[1]}×{img.shape[2]} = {n_pixels:,} 像元")
            
            if not self.is_running:
                return
            
            # 读取类别信息
            self.log(f"\n📊 读取类别信息...")
            class_names, class_colors, _ = self.backend.get_class_info_from_shp(
                self.train_shp_path.get(), 
                self.class_attr.get(), 
                self.name_attr.get()
            )
            self.class_names_dict = class_names
            self.class_colors_dict = class_colors
            self.log(f"   类别: {list(class_names.values())}")
            
            # 提取训练样本
            self.log(f"\n🎯 处理训练样本...")
            self.status_var.set("处理训练样本...")
            train_mask = self.backend.rasterize_samples(
                self.train_shp_path.get(), img, self.class_attr.get()
            )
            
            max_samples = self.max_samples.get() if self.enable_sampling.get() else None
            
            X_train, y_train, n_nan, n_inf, n_sampled = self.backend.extract_samples(
                img, train_mask, 
                ignore_background=self.ignore_background.get(),
                background_value=self.background_value.get(),
                max_samples=max_samples
            )
            
            self.log(f"   训练样本数: {len(y_train):,}")
            if n_nan > 0:
                self.log(f"   └─ 移除NaN: {n_nan:,}")
            if n_sampled > 0:
                self.log(f"   └─ 采样减少: {n_sampled:,}")
            
            if not self.is_running:
                return
            
            # 提取验证样本
            val_exists = os.path.exists(self.val_shp_path.get())
            if val_exists:
                self.log(f"\n✅ 处理验证样本...")
                val_mask = self.backend.rasterize_samples(
                    self.val_shp_path.get(), img, self.class_attr.get()
                )
                
                if self.ignore_background.get():
                    background_mask = self.backend.get_background_mask(
                        img, self.background_value.get()
                    )
                    valid_val = (val_mask > 0) & (~background_mask)
                else:
                    valid_val = val_mask > 0
                
                yv_true = val_mask[valid_val]
                self.log(f"   验证样本数: {len(yv_true):,}")
            
            # 分类器训练和评估
            all_classifiers = self.backend.get_all_classifiers(
                self.n_estimators.get(), 
                fast_mode=self.fast_mode.get(),
                n_train_samples=len(y_train)
            )
            
            comparison_results = []
            total_start_time = time.time()
            best_accuracy = 0
            best_clf_code = None
            
            for i, clf_code in enumerate(selected_classifiers):
                if not self.is_running:
                    break
                
                clf, clf_name, clf_desc, needs_encoding, needs_scaling, speed_tag = all_classifiers[clf_code]
                
                self.log(f"\n{'='*80}")
                self.log(f"[{i+1}/{len(selected_classifiers)}] {clf_name}")
                self.log(f"{'='*80}")
                
                self.status_var.set(f"[{i+1}/{len(selected_classifiers)}] 训练 {clf_name}...")
                
                clf_dir = out_dir / clf_code
                clf_dir.mkdir(exist_ok=True)
                
                try:
                    # 数据预处理
                    label_encoder = None
                    scaler = None
                    X_train_use = X_train.copy()
                    y_train_use = y_train.copy()
                    
                    if needs_encoding:
                        self.log("   🔄 标签编码...")
                        label_encoder = LabelEncoder()
                        y_train_use = label_encoder.fit_transform(y_train)
                    
                    if needs_scaling:
                        self.log("   📏 特征缩放...")
                        scaler = StandardScaler()
                        X_train_use = scaler.fit_transform(X_train_use)
                    
                    # 训练
                    self.log("   🔨 训练中...")
                    train_start = time.time()
                    clf.fit(X_train_use, y_train_use)
                    train_time = time.time() - train_start
                    self.log(f"   ✓ 训练完成: {train_time:.2f}秒")
                    
                    # 训练集精度
                    y_train_pred = clf.predict(X_train_use)
                    
                    if label_encoder is not None:
                        y_train_pred = label_encoder.inverse_transform(y_train_pred)
                    
                    train_metrics = self.backend.calculate_metrics(y_train, y_train_pred)
                    self.log(f"   📈 训练集 - 精度: {train_metrics['overall_accuracy']:.4f}")
                    
                    if not self.is_running:
                        break
                    
                    # 预测整幅影像
                    self.log("   🗺️  预测影像...")
                    self.status_var.set(f"[{i+1}/{len(selected_classifiers)}] 预测 {clf_name}...")
                    
                    pred_start = time.time()
                    classified_path = clf_dir / f"classified_{clf_code}.tif"
                    
                    def update_progress(progress):
                        self.progress_var.set(progress)
                    
                    self.backend.predict_by_block(
                        clf, img, classified_path, 
                        block_size=self.block_size.get(),
                        ignore_background=self.ignore_background.get(),
                        background_value=self.background_value.get(),
                        progress_callback=update_progress,
                        label_encoder=label_encoder,
                        scaler=scaler
                    )
                    
                    pred_time = time.time() - pred_start
                    self.log(f"   ✓ 预测完成: {pred_time:.2f}秒")
                    
                    # 验证集精度
                    val_metrics = {'overall_accuracy': np.nan, 'kappa': np.nan}
                    yv_pred = None
                    
                    if val_exists:
                        with rxr.open_rasterio(classified_path) as pred_img:
                            pred_arr = pred_img.values.squeeze()
                        
                        yv_pred = pred_arr[valid_val]
                        val_metrics = self.backend.calculate_metrics(yv_true, yv_pred)
                        self.log(f"   📊 验证集 - 精度: {val_metrics['overall_accuracy']:.4f}")
                        
                        # 记录最佳分类器
                        if val_metrics['overall_accuracy'] > best_accuracy:
                            best_accuracy = val_metrics['overall_accuracy']
                            best_clf_code = clf_code
                            self.best_result_path = classified_path
                            self.current_y_true = yv_true
                            self.current_y_pred = yv_pred
                    
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
                    }
                    comparison_results.append(result)
                    self.comparison_results = comparison_results
                    
                    # 实时更新图表
                    self.root.after(0, self.update_accuracy_plot)
                    self.root.after(0, self.update_time_plot)
                    
                    self.log(f"   ✅ {clf_name} 完成!")
                    
                except Exception as e:
                    self.log(f"   ❌ {clf_name} 失败: {str(e)}")
                    continue
                
                self.progress_var.set((i + 1) / len(selected_classifiers) * 100)
            
            # 生成报告
            if comparison_results and self.is_running:
                total_time = time.time() - total_start_time
                
                self.log(f"\n{'='*80}")
                self.log("📝 生成报告...")
                
                comparison_df = pd.DataFrame(comparison_results)
                
                # 保存CSV
                comparison_df.to_csv(out_dir / "classifier_comparison.csv", 
                                   index=False, encoding='utf-8-sig')
                
                # 导出Excel
                self.export_to_excel(out_dir)
                
                # 文字报告
                with open(out_dir / "comparison_summary.txt", 'w', encoding='utf-8') as f:
                    f.write("遥感影像分类器性能对比报告\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"影像: {img.shape[1]}×{img.shape[2]}\n")
                    f.write(f"训练样本: {len(y_train):,}\n")
                    f.write(f"成功: {len(comparison_results)}/{len(selected_classifiers)}\n")
                    f.write(f"总耗时: {total_time/60:.1f} 分钟\n\n")
                    
                    sorted_df = comparison_df.sort_values('验证集精度', ascending=False)
                    f.write("验证集精度排名:\n")
                    f.write("-"*70 + "\n")
                    for idx, (_, row) in enumerate(sorted_df.iterrows(), 1):
                        f.write(f"{idx}. {row['分类器名称']:15s} - "
                               f"精度: {row['验证集精度']:.4f}\n")
                
                # 更新混淆矩阵
                if self.current_y_true is not None and self.current_y_pred is not None:
                    val_classes = sorted(np.unique(self.current_y_true))
                    val_class_names = [class_names.get(c, f'类别_{c}') for c in val_classes]
                    self.root.after(0, lambda: self.update_confusion_matrix(
                        self.current_y_true, self.current_y_pred, val_class_names
                    ))
                
                # 更新结果预览
                if self.best_result_path:
                    self.root.after(0, lambda: self.update_result_preview(
                        self.image_path.get(), self.best_result_path, 
                        class_names, class_colors
                    ))
                
                self.log("✅ 所有任务完成!")
                self.log(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
                
                best_clf = comparison_df.loc[comparison_df['验证集精度'].idxmax()]
                self.log(f"\n🏆 最佳: {best_clf['分类器名称']} ({best_clf['验证集精度']:.4f})")
                
                self.status_var.set(f"✅ 完成! 最佳: {best_clf['分类器名称']}")
                
                # 切换到精度对比标签页
                self.root.after(0, lambda: self.notebook.select(1))
                
                messagebox.showinfo("任务完成", 
                    f"🎉 分类任务完成!\n\n"
                    f"✅ 成功: {len(comparison_results)}/{len(selected_classifiers)}\n"
                    f"🏆 最佳: {best_clf['分类器名称']} ({best_clf['验证集精度']:.4f})\n"
                    f"📊 结果已导出为Excel和CSV")
            
        except Exception as e:
            self.log(f"\n❌ 错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("错误", f"发生错误:\n{str(e)}")
            self.status_var.set("❌ 错误")
        
        finally:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
            self.is_running = False
    
    def open_result_dir(self):
        """打开结果目录"""
        out_dir = Path(self.output_dir.get())
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
            messagebox.showwarning("警告", "结果目录不存在！")

# ==================== 主程序入口 ====================
def main():
    """程序入口"""
    print("="*80)
    print("  遥感影像监督分类系统 v4.1 - 专业版")
    print("="*80)
    print("\n正在检查依赖库...")
    
    root = tk.Tk()
    app = ClassificationGUI(root)
    
    # 欢迎信息
    app.log("="*80)
    app.log("  遥感影像监督分类系统 v4.1 - 专业版")
    app.log("="*80)
    app.log("\n主要特性:")
    app.log("  ✓ 自定义背景值输入")
    app.log("  ✓ 字段下拉框自动识别")
    app.log("  ✓ 实时精度对比图表")
    app.log("  ✓ 混淆矩阵可视化")
    app.log("  ✓ 分类结果预览")
    app.log("  ✓ Excel格式报告导出")
    app.log("\n使用流程:")
    app.log("  1. 选择影像和样本文件")
    app.log("  2. 点击'刷新字段列表'选择类别字段")
    app.log("  3. 设置背景值和其他参数")
    app.log("  4. 选择分类器")
    app.log("  5. 点击'开始分类'")
    app.log("  6. 查看右侧实时图表")
    app.log("="*80)
    app.log("")
    
    print("\n✓ 系统启动成功!")
    
    root.mainloop()

if __name__ == "__main__":
    main()
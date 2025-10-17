#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遥感影像监督分类系统 - Streamlit Web版 v4.2
==========================================
修复多分类器循环执行问题
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from pathlib import Path
import tempfile
import base64
import zipfile
import shutil
import gc

sns.set_style("whitegrid")

# 设置matplotlib中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 检查可选库
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    st.warning("⚠️ 未安装openpyxl，将无法导出Excel文件")

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
        except Exception:
            pass
        
        try:
            import lightgbm
            from lightgbm import LGBMClassifier
            _ = LGBMClassifier(n_estimators=10, verbose=-1)
            self.has_lightgbm = True
        except Exception:
            pass
    
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
            st.error(f"读取shapefile字段失败: {e}")
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
        
        # 检查样本数量
        if len(y) == 0:
            return X, y, n_nan, n_inf, 0
        
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
                progress_callback((i + 1) / total_blocks)
        
        # 保存结果
        prediction_da = xr.DataArray(prediction, dims=['y', 'x'],
                                     coords={'y': image.coords['y'], 'x': image.coords['x']})
        
        prediction_da.rio.write_crs(image.rio.crs, inplace=True)
        prediction_da.rio.write_transform(image.rio.transform(), inplace=True)
        prediction_da.rio.write_nodata(background_value, inplace=True)
        
        prediction_da.rio.to_raster(out_path, driver='GTiff', dtype='uint16', 
                                    compress='lzw', tiled=True)
        return out_path

# ==================== 辅助函数 ====================
def extract_zip_file(zip_file, extract_dir):
    """解压ZIP文件到指定目录"""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 查找.shp文件
    shp_files = list(Path(extract_dir).glob("*.shp"))
    if not shp_files:
        raise ValueError("ZIP文件中未找到.shp文件")
    
    return str(shp_files[0])

def save_uploaded_file(uploaded_file, temp_dir, filename):
    """保存上传的文件到临时目录"""
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def safe_delete_temp_dir(temp_dir):
    """安全删除临时目录，处理文件占用问题"""
    if not temp_dir or not os.path.exists(temp_dir):
        return
    
    max_retries = 3
    for i in range(max_retries):
        try:
            # 先尝试正常删除
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if i < max_retries - 1:
                # 等待一段时间后重试
                time.sleep(1)
                # 强制垃圾回收
                gc.collect()
            else:
                # 最后一次尝试，忽略错误
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

def add_log(message):
    """添加日志消息"""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    st.session_state.log_messages.append(message)
    # 保持日志长度合理
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-50:]

# ==================== Streamlit应用 ====================
def main():
    st.set_page_config(
        page_title="遥感影像监督分类系统 v4.2 - Web版",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化session state
    if 'backend' not in st.session_state:
        st.session_state.backend = ClassificationBackend()
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = []
    
    if 'comparison_df' not in st.session_state:
        st.session_state.comparison_df = None
    
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    
    # 新增：分类器执行状态
    if 'current_classifier_index' not in st.session_state:
        st.session_state.current_classifier_index = 0
    
    if 'selected_classifiers_list' not in st.session_state:
        st.session_state.selected_classifiers_list = []
    
    if 'classification_params' not in st.session_state:
        st.session_state.classification_params = {}
    
    # 标题和介绍
    st.title("🛰️ 遥感影像监督分类系统 v4.2 - Web版")
    st.markdown("""
    ### 专业级遥感影像分类工具
    
    基于机器学习的多算法对比分类系统，支持15+种分类器，提供完整的精度评估和可视化分析。
    
    **注意**: 
    - Shapefile请打包为ZIP格式上传，包含.shp, .shx, .dbf, .prj等所有必需文件
    - 确保训练样本与影像文件在相同的地理坐标系下
    - 如果遇到样本提取问题，请检查背景值设置是否正确
    """)
    
    # 侧边栏 - 参数设置
    st.sidebar.header("📋 参数设置")
    
    # 文件上传
    st.sidebar.subheader("📁 数据文件")
    image_file = st.sidebar.file_uploader("遥感影像文件", type=['tif', 'tiff'], key="image")
    train_zip_file = st.sidebar.file_uploader("训练样本ZIP文件", type=['zip'], key="train_zip",
                                             help="请上传包含.shp, .shx, .dbf, .prj等文件的ZIP压缩包")
    
    val_zip_file = st.sidebar.file_uploader("验证样本ZIP文件 (可选)", type=['zip'], key="val_zip",
                                           help="请上传包含.shp, .shx, .dbf, .prj等文件的ZIP压缩包")
    
    # 字段配置
    st.sidebar.subheader("🏷️ 字段配置")
    
    # 如果上传了训练样本ZIP文件，处理并获取字段
    if train_zip_file and 'train_shp_path' not in st.session_state:
        with st.spinner("正在解压训练样本文件..."):
            try:
                # 创建临时目录
                temp_dir = tempfile.mkdtemp()
                train_zip_path = save_uploaded_file(train_zip_file, temp_dir, "train_samples.zip")
                
                # 解压ZIP文件
                extract_dir = os.path.join(temp_dir, "train_extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                train_shp_path = extract_zip_file(train_zip_path, extract_dir)
                st.session_state.train_shp_path = train_shp_path
                st.session_state.train_temp_dir = temp_dir
                
                st.sidebar.success("训练样本解压成功!")
                
            except Exception as e:
                st.sidebar.error(f"解压训练样本失败: {str(e)}")
    
    # 如果上传了验证样本ZIP文件，处理并获取字段
    if val_zip_file and 'val_shp_path' not in st.session_state:
        with st.spinner("正在解压验证样本文件..."):
            try:
                # 创建临时目录
                temp_dir = tempfile.mkdtemp()
                val_zip_path = save_uploaded_file(val_zip_file, temp_dir, "val_samples.zip")
                
                # 解压ZIP文件
                extract_dir = os.path.join(temp_dir, "val_extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                val_shp_path = extract_zip_file(val_zip_path, extract_dir)
                st.session_state.val_shp_path = val_shp_path
                st.session_state.val_temp_dir = temp_dir
                
                st.sidebar.success("验证样本解压成功!")
                
            except Exception as e:
                st.sidebar.error(f"解压验证样本失败: {str(e)}")
    
    # 获取字段列表
    if 'train_shp_path' in st.session_state:
        fields = st.session_state.backend.get_shapefile_fields(st.session_state.train_shp_path)
        if fields:
            fields = [f for f in fields if f.lower() != 'geometry']
            
            class_attr = st.sidebar.selectbox("类别编号字段", fields, key="class_attr")
            name_attr = st.sidebar.selectbox("类别名称字段", fields, key="name_attr")
        else:
            st.sidebar.error("无法读取shapefile字段")
            class_attr = st.sidebar.text_input("类别编号字段", key="class_attr")
            name_attr = st.sidebar.text_input("类别名称字段", key="name_attr")
    else:
        class_attr = st.sidebar.text_input("类别编号字段", key="class_attr")
        name_attr = st.sidebar.text_input("类别名称字段", key="name_attr")
    
    # 背景值设置
    st.sidebar.subheader("🎨 背景值设置")
    ignore_background = st.sidebar.checkbox("忽略背景值", value=True, key="ignore_bg")
    background_value = st.sidebar.number_input("背景值", value=0, key="bg_value")
    
    # 分类参数
    st.sidebar.subheader("⚙️ 分类参数")
    n_estimators = st.sidebar.slider("树模型数量", 10, 500, 100, key="n_estimators")
    block_size = st.sidebar.selectbox("分块大小", [256, 512, 1024, 2048], index=1, key="block_size")
    
    # 性能优化
    st.sidebar.subheader("⚡ 性能优化")
    enable_sampling = st.sidebar.checkbox("启用采样", value=True, key="enable_sampling")
    max_samples = st.sidebar.slider("最大样本数", 10000, 200000, 50000, step=10000, key="max_samples")
    fast_mode = st.sidebar.checkbox("快速模式", value=False, key="fast_mode")
    
    # 分类器选择
    st.sidebar.subheader("🤖 分类器选择")
    all_classifiers = st.session_state.backend.get_all_classifiers()
    
    # 分类器分组
    svm_codes = ["svm_linear", "linear_svc", "sgd_svm", "nystroem_svm", "rbf_sampler_svm", "svm_rbf"]
    tree_codes = ["rf", "et", "dt", "xgb", "lgb", "gb", "ada"]
    other_codes = ["knn", "nb", "lr", "mlp"]
    
    selected_classifiers = []
    
    # SVM分类器
    st.sidebar.markdown("**SVM分类器:**")
    for code in svm_codes:
        if code in all_classifiers:
            _, name, _, _, _, _ = all_classifiers[code]
            if st.sidebar.checkbox(name, key=f"clf_{code}"):
                selected_classifiers.append(code)
    
    # 树模型分类器
    st.sidebar.markdown("**树模型分类器:**")
    for code in tree_codes:
        if code in all_classifiers:
            _, name, _, _, _, _ = all_classifiers[code]
            if st.sidebar.checkbox(name, key=f"clf_{code}"):
                selected_classifiers.append(code)
    
    # 其他分类器
    st.sidebar.markdown("**其他分类器:**")
    for code in other_codes:
        if code in all_classifiers:
            _, name, _, _, _, _ = all_classifiers[code]
            if st.sidebar.checkbox(name, key=f"clf_{code}"):
                selected_classifiers.append(code)
    
    # 快捷选择按钮
    st.sidebar.subheader("快捷选择")
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("全选", key="select_all"):
            for code in all_classifiers.keys():
                st.session_state[f"clf_{code}"] = True
    
    with col2:
        if st.button("推荐", key="select_recommended"):
            recommended = ["rf", "xgb", "et", "lgb", "linear_svc", "nystroem_svm"]
            for code in all_classifiers.keys():
                st.session_state[f"clf_{code}"] = (code in recommended)
    
    with col3:
        if st.button("快速", key="select_fast"):
            fast = ["rf", "et", "dt", "xgb", "lgb", "nb", "lr", "sgd_svm", "linear_svc"]
            for code in all_classifiers.keys():
                st.session_state[f"clf_{code}"] = (code in fast)
    
    # 主内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 运行控制", "📊 精度对比", "🔥 混淆矩阵", "⏱️ 时间对比", "🗺️ 结果预览"])
    
    with tab1:
        st.header("运行控制")
        
        # 运行按钮
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("▶ 开始分类", type="primary", disabled=st.session_state.is_running):
                if not image_file:
                    st.error("请选择影像文件！")
                elif 'train_shp_path' not in st.session_state:
                    st.error("请上传训练样本ZIP文件！")
                elif not class_attr:
                    st.error("请选择类别编号字段！")
                elif not selected_classifiers:
                    st.error("请至少选择一个分类器！")
                else:
                    # 保存分类参数
                    st.session_state.classification_params = {
                        'image_file': image_file,
                        'class_attr': class_attr,
                        'name_attr': name_attr,
                        'ignore_background': ignore_background,
                        'background_value': background_value,
                        'n_estimators': n_estimators,
                        'block_size': block_size,
                        'enable_sampling': enable_sampling,
                        'max_samples': max_samples,
                        'fast_mode': fast_mode,
                        'selected_classifiers': selected_classifiers
                    }
                    
                    st.session_state.is_running = True
                    st.session_state.log_messages = []
                    st.session_state.comparison_results = []
                    st.session_state.comparison_df = None
                    st.session_state.current_classifier_index = 0
                    st.session_state.selected_classifiers_list = selected_classifiers.copy()
                    
                    add_log("🚀 开始分类任务...")
                    add_log(f"📋 选择的分类器: {', '.join([all_classifiers[code][1] for code in selected_classifiers])}")
        
        with col2:
            if st.button("⏸ 停止", disabled=not st.session_state.is_running):
                st.session_state.is_running = False
                add_log("⏹️ 用户手动停止分类任务")
        
        with col3:
            if st.session_state.comparison_df is not None and not st.session_state.comparison_df.empty:
                if st.button("📥 下载结果"):
                    download_results()
        
        # 进度显示
        if st.session_state.progress > 0:
            st.progress(st.session_state.progress)
            if (st.session_state.current_classifier_index > 0 and 
                st.session_state.selected_classifiers_list):
                total_classifiers = len(st.session_state.selected_classifiers_list)
                current_index = st.session_state.current_classifier_index
                current_classifier = st.session_state.selected_classifiers_list[current_index - 1]
                all_classifiers = st.session_state.backend.get_all_classifiers()
                classifier_name = all_classifiers[current_classifier][1] if current_classifier in all_classifiers else current_classifier
                st.write(f"正在处理: {classifier_name} ({current_index}/{total_classifiers})")
        
        # 日志显示
        st.subheader("运行日志")
        log_container = st.container()
        with log_container:
            for message in st.session_state.log_messages:
                st.text(message)
        
        # 如果正在运行，执行分类任务
        if st.session_state.is_running and st.session_state.selected_classifiers_list:
            if st.session_state.current_classifier_index < len(st.session_state.selected_classifiers_list):
                # 执行下一个分类器
                run_classification_step()
            else:
                # 所有分类器完成
                st.session_state.is_running = False
                add_log("✅ 所有分类器处理完成!")
                st.success("所有分类器处理完成!")
    
    with tab2:
        st.header("精度对比")
        if st.session_state.comparison_df is not None and not st.session_state.comparison_df.empty:
            display_accuracy_comparison()
        else:
            st.info("暂无分类结果，请先运行分类")
    
    with tab3:
        st.header("混淆矩阵")
        if (st.session_state.comparison_df is not None and not st.session_state.comparison_df.empty and 
            'best_confusion_matrix' in st.session_state and st.session_state.best_confusion_matrix is not None):
            display_confusion_matrix()
        else:
            st.info("暂无混淆矩阵数据")
    
    with tab4:
        st.header("时间对比")
        if st.session_state.comparison_df is not None and not st.session_state.comparison_df.empty:
            display_time_comparison()
        else:
            st.info("暂无时间对比数据")
    
    with tab5:
        st.header("分类结果预览")
        if (st.session_state.comparison_df is not None and not st.session_state.comparison_df.empty and 
            'best_result_path' in st.session_state and st.session_state.best_result_path is not None):
            display_result_preview()
        else:
            st.info("暂无分类结果预览")
    
    # 清理临时文件
    if st.sidebar.button("清理临时文件"):
        if 'train_temp_dir' in st.session_state:
            safe_delete_temp_dir(st.session_state.train_temp_dir)
            st.session_state.train_temp_dir = None
            st.session_state.train_shp_path = None
        
        if 'val_temp_dir' in st.session_state:
            safe_delete_temp_dir(st.session_state.val_temp_dir)
            st.session_state.val_temp_dir = None
            st.session_state.val_shp_path = None
        
        st.sidebar.success("临时文件已清理!")

def run_classification_step():
    """执行单个分类器的分类任务"""
    
    backend = st.session_state.backend
    params = st.session_state.classification_params
    selected_classifiers = st.session_state.selected_classifiers_list
    current_index = st.session_state.current_classifier_index
    
    if current_index >= len(selected_classifiers):
        return
    
    clf_code = selected_classifiers[current_index]
    all_classifiers = backend.get_all_classifiers()
    
    if clf_code not in all_classifiers:
        add_log(f"❌ 未知的分类器代码: {clf_code}")
        st.session_state.current_classifier_index += 1
        st.rerun()
        return
    
    clf, clf_name, clf_desc, needs_encoding, needs_scaling, speed_tag = all_classifiers[clf_code]
    
    add_log(f"\n{'='*80}")
    add_log(f"[{current_index+1}/{len(selected_classifiers)}] {clf_name}")
    add_log(f"{'='*80}")
    
    # 如果是第一次运行，初始化临时目录和数据
    if current_index == 0:
        # 创建主临时目录
        main_temp_dir = tempfile.mkdtemp()
        st.session_state.main_temp_dir = main_temp_dir
        
        try:
            temp_path = Path(main_temp_dir)
            
            # 保存上传的文件
            image_path = temp_path / "image.tif"
            with open(image_path, 'wb') as f:
                f.write(params['image_file'].getvalue())
            
            # 使用session state中保存的shapefile路径
            train_shp_path = st.session_state.train_shp_path
            val_shp_path = st.session_state.val_shp_path if 'val_shp_path' in st.session_state else None
            
            # 读取影像
            add_log("📁 读取影像...")
            try:
                # 使用上下文管理器确保文件正确关闭
                with rxr.open_rasterio(image_path, masked=True) as img:
                    # 将数据加载到内存中
                    img_data = img.load()
                    n_pixels = img_data.shape[1] * img_data.shape[2]
                    add_log(f"   尺寸: {img_data.shape[1]}×{img_data.shape[2]} = {n_pixels:,} 像元")
                    add_log(f"   波段数: {img_data.shape[0]}")
                    add_log(f"   数据类型: {img_data.dtype}")
                    add_log(f"   坐标系统: {img_data.rio.crs}")
                    
                    # 读取类别信息
                    add_log("📊 读取类别信息...")
                    class_names, class_colors, _ = backend.get_class_info_from_shp(
                        train_shp_path, params['class_attr'], params['name_attr']
                    )
                    st.session_state.class_names = class_names
                    st.session_state.class_colors = class_colors
                    add_log(f"   类别: {list(class_names.values())}")
                    
                    # 提取训练样本
                    add_log("🎯 处理训练样本...")
                    train_mask = backend.rasterize_samples(train_shp_path, img_data, params['class_attr'])
                    
                    max_samples_val = params['max_samples'] if params['enable_sampling'] else None
                    
                    X_train, y_train, n_nan, n_inf, n_sampled = backend.extract_samples(
                        img_data, train_mask, 
                        ignore_background=params['ignore_background'],
                        background_value=params['background_value'],
                        max_samples=max_samples_val
                    )
                    
                    add_log(f"   训练样本数: {len(y_train):,}")
                    if n_nan > 0:
                        add_log(f"   └─ 移除NaN: {n_nan:,}")
                    if n_inf > 0:
                        add_log(f"   └─ 移除Inf: {n_inf:,}")
                    if n_sampled > 0:
                        add_log(f"   └─ 采样减少: {n_sampled:,}")
                    
                    # 检查样本数量
                    if len(y_train) == 0:
                        add_log("❌ 错误: 没有提取到训练样本!")
                        st.session_state.is_running = False
                        return
                    
                    # 提取验证样本
                    val_exists = val_shp_path and os.path.exists(val_shp_path)
                    yv_true = None
                    valid_val = None
                    if val_exists:
                        add_log("✅ 处理验证样本...")
                        val_mask = backend.rasterize_samples(val_shp_path, img_data, params['class_attr'])
                        
                        if params['ignore_background']:
                            background_mask = backend.get_background_mask(img_data, params['background_value'])
                            valid_val = (val_mask > 0) & (~background_mask)
                        else:
                            valid_val = val_mask > 0
                        
                        yv_true = val_mask[valid_val]
                        add_log(f"   验证样本数: {len(yv_true):,}")
                    
                    # 保存数据到session state
                    st.session_state.temp_path = temp_path
                    st.session_state.image_path = image_path
                    st.session_state.img_data = img_data
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    st.session_state.val_exists = val_exists
                    st.session_state.yv_true = yv_true
                    st.session_state.valid_val = valid_val
                    
            except Exception as e:
                add_log(f"❌ 影像读取错误: {str(e)}")
                st.session_state.is_running = False
                return
            
        except Exception as e:
            add_log(f"\n❌ 错误: {str(e)}")
            st.session_state.is_running = False
            return
    
    # 处理当前分类器
    try:
        temp_path = st.session_state.temp_path
        img_data = st.session_state.img_data
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        val_exists = st.session_state.val_exists
        yv_true = st.session_state.yv_true
        valid_val = st.session_state.valid_val
        
        clf_dir = temp_path / clf_code
        clf_dir.mkdir(exist_ok=True)
        
        # 数据预处理
        label_encoder = None
        scaler = None
        X_train_use = X_train.copy()
        y_train_use = y_train.copy()
        
        if needs_encoding:
            add_log("   🔄 标签编码...")
            label_encoder = LabelEncoder()
            y_train_use = label_encoder.fit_transform(y_train)
        
        if needs_scaling:
            add_log("   📏 特征缩放...")
            scaler = StandardScaler()
            X_train_use = scaler.fit_transform(X_train_use)
        
        # 训练
        add_log("   🔨 训练中...")
        train_start = time.time()
        clf.fit(X_train_use, y_train_use)
        train_time = time.time() - train_start
        add_log(f"   ✓ 训练完成: {train_time:.2f}秒")
        
        # 训练集精度
        y_train_pred = clf.predict(X_train_use)
        
        if label_encoder is not None:
            y_train_pred = label_encoder.inverse_transform(y_train_pred)
        
        train_metrics = backend.calculate_metrics(y_train, y_train_pred)
        add_log(f"   📈 训练集 - 精度: {train_metrics['overall_accuracy']:.4f}")
        
        if not st.session_state.is_running:
            return
        
        # 预测整幅影像
        add_log("   🗺️  预测影像...")
        
        def update_progress(progress):
            st.session_state.progress = progress
        
        pred_start = time.time()
        classified_path = clf_dir / f"classified_{clf_code}.tif"
        
        backend.predict_by_block(
            clf, img_data, classified_path, 
            block_size=params['block_size'],
            ignore_background=params['ignore_background'],
            background_value=params['background_value'],
            progress_callback=update_progress,
            label_encoder=label_encoder,
            scaler=scaler
        )
        
        pred_time = time.time() - pred_start
        add_log(f"   ✓ 预测完成: {pred_time:.2f}秒")
        
        # 验证集精度
        val_metrics = {'overall_accuracy': np.nan, 'kappa': np.nan}
        yv_pred = None
        
        if val_exists and yv_true is not None:
            # 使用上下文管理器读取预测结果
            with rxr.open_rasterio(classified_path) as pred_img:
                pred_arr = pred_img.values.squeeze()
            
            yv_pred = pred_arr[valid_val]
            val_metrics = backend.calculate_metrics(yv_true, yv_pred)
            add_log(f"   📊 验证集 - 精度: {val_metrics['overall_accuracy']:.4f}")
            
            # 记录最佳分类器
            if ('best_accuracy' not in st.session_state or 
                val_metrics['overall_accuracy'] > st.session_state.best_accuracy):
                st.session_state.best_accuracy = val_metrics['overall_accuracy']
                st.session_state.best_result_path = classified_path
                
                # 保存混淆矩阵数据
                cm = confusion_matrix(yv_true, yv_pred)
                st.session_state.best_confusion_matrix = cm
                st.session_state.best_y_true = yv_true
                st.session_state.best_y_pred = yv_pred
        
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
        
        # 更新比较结果
        if st.session_state.comparison_results is None:
            st.session_state.comparison_results = []
        
        st.session_state.comparison_results.append(result)
        st.session_state.comparison_df = pd.DataFrame(st.session_state.comparison_results)
        
        add_log(f"   ✅ {clf_name} 完成!")
        
    except Exception as e:
        add_log(f"   ❌ {clf_name} 失败: {str(e)}")
        import traceback
        add_log(traceback.format_exc())
    
    # 更新进度并继续下一个分类器
    st.session_state.current_classifier_index += 1
    st.session_state.progress = st.session_state.current_classifier_index / len(selected_classifiers)
    
    # 自动刷新界面
    st.rerun()

# 其他显示函数保持不变...
def display_accuracy_comparison():
    """显示精度对比"""
    if st.session_state.comparison_df is None or st.session_state.comparison_df.empty:
        st.info("暂无分类结果，请先运行分类")
        return
    
    df = st.session_state.comparison_df
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 显示详细数据
    st.subheader("详细结果")
    st.dataframe(df)

def display_confusion_matrix():
    """显示混淆矩阵"""
    if (st.session_state.comparison_df is None or st.session_state.comparison_df.empty or 
        'best_confusion_matrix' not in st.session_state or st.session_state.best_confusion_matrix is None):
        st.info("暂无混淆矩阵数据")
        return
    
    cm = st.session_state.best_confusion_matrix
    class_names = list(st.session_state.class_names.values())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量'}, ax=ax)
    
    ax.set_xlabel('预测类别', fontsize=11)
    ax.set_ylabel('真实类别', fontsize=11)
    ax.set_title('最佳分类器混淆矩阵', fontsize=12, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)

def display_time_comparison():
    """显示时间对比"""
    if st.session_state.comparison_df is None or st.session_state.comparison_df.empty:
        st.info("暂无时间对比数据")
        return
    
    df = st.session_state.comparison_df
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
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
    
    plt.tight_layout()
    st.pyplot(fig)

def display_result_preview():
    """显示分类结果预览"""
    if (st.session_state.comparison_df is None or st.session_state.comparison_df.empty or 
        'best_result_path' not in st.session_state or st.session_state.best_result_path is None):
        st.info("暂无分类结果预览")
        return
    
    try:
        best_result_path = st.session_state.best_result_path
        class_names = st.session_state.class_names
        class_colors = st.session_state.class_colors
        
        # 这里简化处理，实际应用中需要处理影像显示
        st.info("分类结果预览功能在Web版本中受限")
        st.write(f"最佳分类结果文件: {best_result_path.name}")
        st.write("类别信息:", class_names)
        
        # 在实际应用中，这里应该显示分类结果的缩略图
        # 但由于Web环境限制，我们只显示基本信息
        
    except Exception as e:
        st.error(f"预览显示错误: {str(e)}")

def download_results():
    """下载结果文件"""
    if st.session_state.comparison_df is None or st.session_state.comparison_df.empty:
        st.error("没有可下载的结果")
        return
    
    # 创建下载文件
    csv = st.session_state.comparison_df.to_csv(index=False)
    
    st.download_button(
        label="📥 下载CSV结果",
        data=csv,
        file_name="classification_results.csv",
        mime="text/csv"
    )
    
    # 生成报告文本
    report = f"""遥感影像分类器性能对比报告
================================
时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
训练样本: {len(st.session_state.y_train):,}
总耗时: 请查看具体分类器时间

验证集精度排名:
"""
    
    sorted_df = st.session_state.comparison_df.sort_values('验证集精度', ascending=False)
    for idx, (_, row) in enumerate(sorted_df.iterrows(), 1):
        report += f"{idx}. {row['分类器名称']:15s} - 精度: {row['验证集精度']:.4f}\n"
    
    st.download_button(
        label="📥 下载文本报告",
        data=report,
        file_name="classification_report.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
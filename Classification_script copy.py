#!/usr/bin/env python3
# coding: utf-8
"""
优化版随机森林分类脚本
-------------------------------------------------
主要功能包括：
1. 利用 rasterio + geopandas 对遥感影像进行监督分类；
2. 将训练与验证矢量样本栅格化，以匹配影像像元；
3. 使用随机森林（Random Forest）算法进行训练；
4. 采用分块（block）方式对大影像进行预测，节约内存；
5. 输出分类结果 GeoTIFF；
6. 生成文字报告与混淆矩阵图。
"""

import os
import sys
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示（MacOS）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==========================================================
# -------- 用户配置区（可根据需要修改）---------------------
# ==========================================================

EST = 500  # 随机森林的树数量，即弱分类器数量
N_JOBS = -1  # 并行线程数量，-1 表示使用所有CPU核心
IMG_RS = r'D:\code313\Geo_programe\rasterio\RF\data\2017_09_05_stack.tif'  # 输入影像路径
TRAIN_SHP = r'D:\code313\Geo_programe\rasterio\RF\data\cal.shp'  # 训练样本矢量路径
VAL_SHP = r'D:\code313\Geo_programe\rasterio\RF\data\val.shp'  # 验证样本矢量路径
ATTRIBUTE = 'class'  # 矢量文件中表示分类类别的属性字段名

# 输出文件路径设置
out_dir = os.path.dirname(IMG_RS)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

CLASSIFICATION_IMAGE = os.path.join(out_dir, 'results\\HueMo2018_14bands_class_optimized.tif')
RESULTS_TXT = os.path.join(out_dir, 'results_txt_optimized.txt')
BLOCK_HEIGHT = 512  # 分块处理的窗口高度，影响内存占用
USE_SCALER = False  # 是否对特征进行标准化处理
CLASS_WEIGHT = None  # 类别权重，可设为 'balanced' 以处理样本不均衡问题


# ==========================================================
# -------- 初始化日志记录系统 -------------------------------
# ==========================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('rf_class')


# ==========================================================
# -------- 辅助函数定义区 ---------------------------------
# ==========================================================

def write_report(text):
    """将文字信息写入结果报告文件"""
    with open(RESULTS_TXT, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def rasterize_shapefile(shp_path, reference_raster_path, attribute, dtype=np.uint16, all_touched=True):
    """
    将矢量文件（如训练样本）栅格化为与参考影像对齐的数组。
    参数：
        shp_path: 矢量文件路径；
        reference_raster_path: 参考影像路径；
        attribute: 分类字段；
        dtype: 输出数据类型；
        all_touched: 是否栅格化所有接触像元（True 更细致）。
    返回：
        栅格化后的数组（与影像行列一致）。
    """
    gdf = gpd.read_file(shp_path)
    if attribute not in gdf.columns:
        raise ValueError(f"属性 '{attribute}' 未在 {shp_path} 中找到。可用字段: {list(gdf.columns)}")

    # 生成 (geometry, value) 对，用于 rasterize
    shapes = ((geom, int(val)) for geom, val in zip(gdf.geometry, gdf[attribute]))
    with rasterio.open(reference_raster_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)
        arr = rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype=dtype
        )
    return arr


def read_image_bands(raster_path):
    """读取多波段影像，返回三维数组 (bands, rows, cols)。"""
    with rasterio.open(raster_path) as src:
        bands = src.count
        rows = src.height
        cols = src.width
        meta = src.meta.copy()
        data = src.read()
    return data, meta


def sample_training_pixels(img_bands, roi_mask):
    """
    根据训练掩膜提取训练样本像元值。
    参数：
        img_bands: 影像数组 (bands, rows, cols)
        roi_mask: 训练掩膜数组（0为背景，>0为类别）
    返回：
        X: 特征矩阵 (n_samples, n_bands)
        y: 标签数组 (n_samples,)
    """
    mask_idx = np.where(roi_mask > 0)
    if len(mask_idx[0]) == 0:
        raise ValueError("ROI 掩膜中未找到有效训练像元。")

    bands_swapped = np.moveaxis(img_bands, 0, -1)  # 调整维度为 (rows, cols, bands)
    X = bands_swapped[mask_idx]
    y = roi_mask[mask_idx]
    return X, y


def train_random_forest(X, y, n_estimators=EST, n_jobs=N_JOBS, class_weight=CLASS_WEIGHT):
    """
    训练随机森林分类器。
    可选使用标准化；返回模型与标准化器。
    """
    scaler = None
    if USE_SCALER:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        n_jobs=n_jobs,
        verbose=1,
        class_weight=class_weight
    )
    rf.fit(X, y)
    return rf, scaler


def predict_blocked(rf, scaler, src_path, out_path, meta, nodata=0, block_height=BLOCK_HEIGHT):
    """
    采用分块（block）读取影像并逐块预测，以避免内存溢出。
    每次读取一定数量的行，对应 block_height；预测后写入 GeoTIFF。
    """
    out_meta = meta.copy()
    out_meta.update(count=1, dtype=rasterio.uint16, compress='lzw')

    with rasterio.open(src_path) as src, rasterio.open(out_path, 'w', **out_meta) as dst:
        bands = src.count
        height = src.height
        width = src.width

        for row_off in tqdm(range(0, height, block_height), desc='Predicting blocks'):
            h = min(block_height, height - row_off)
            window = Window(0, row_off, width, h)
            block = src.read(window=window)

            b_swapped = np.moveaxis(block, 0, -1)
            reshaped = b_swapped.reshape(-1, bands)
            reshaped = np.nan_to_num(reshaped)
            if scaler is not None:
                reshaped = scaler.transform(reshaped)

            preds = rf.predict(reshaped)
            preds = preds.reshape(h, width).astype(np.uint16)

            # 若输入为nodata，则在输出中置为0
            if src.nodata is not None:
                mask = np.all(block == src.nodata, axis=0) if block.ndim == 3 else (block[0] == src.nodata)
                preds[mask] = 0

            dst.write(preds, 1, window=window)
    return out_path


# ==========================================================
# -------- 主函数 ------------------------------------------
# ==========================================================

def main():
    logger.info("开始随机森林分类流程")
    write_report("Random Forest Classification")
    write_report(f"处理时间: {datetime.datetime.now()}")
    write_report(f"输入影像: {IMG_RS}")
    write_report(f"训练样本: {TRAIN_SHP}")
    write_report(f"验证样本: {VAL_SHP}")
    write_report(f"分类字段: {ATTRIBUTE}")
    write_report(f"输出影像: {CLASSIFICATION_IMAGE}")
    write_report("-------------------------------------------------")

    with rasterio.open(IMG_RS) as src:
        meta = src.meta.copy()
        rows, cols = src.height, src.width
        bands = src.count
        nodata = src.nodata
    logger.info(f"影像大小: {rows}x{cols}, 波段数: {bands}")

    # 栅格化训练样本
    logger.info("正在栅格化训练样本...")
    roi = rasterize_shapefile(TRAIN_SHP, IMG_RS, ATTRIBUTE)
    n_samples = np.count_nonzero(roi)
    logger.info(f"训练样本数量: {n_samples}")
    write_report(f"{n_samples} 个训练样本")

    # 读取影像并提取训练样本特征
    with rasterio.open(IMG_RS) as src:
        img_bands = src.read()
    X, y = sample_training_pixels(img_bands, roi)
    write_report(f"训练类别: {np.unique(y)}")

    X = np.nan_to_num(X)

    # 训练随机森林
    logger.info("正在训练随机森林模型...")
    rf, scaler = train_random_forest(X, y)
    write_report(f"OOB 得分: {rf.oob_score_:.4f}")
    logger.info(f"OOB 得分: {rf.oob_score_:.4f}")

    # 输出特征重要性
    for i, imp in enumerate(rf.feature_importances_, start=1):
        write_report(f"波段 {i} 重要性: {imp:.6f}")

    # 训练集混淆矩阵
    pred_train = rf.predict(X if scaler is None else scaler.transform(X))
    df = pd.crosstab(pd.Series(y, name='truth'), pd.Series(pred_train, name='pred'))
    write_report(str(df))

    # 绘制混淆矩阵图
    cm = confusion_matrix(y, pred_train)
    plt.figure(figsize=(8,6))
    sn.heatmap(cm, annot=True, fmt='g')
    plt.title('训练集混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.tight_layout()
    plt.savefig(os.path.splitext(CLASSIFICATION_IMAGE)[0] + "_cm_train.png")
    plt.close()

    # 分块预测整幅影像
    logger.info("开始分块预测影像...")
    Path(os.path.dirname(CLASSIFICATION_IMAGE)).mkdir(parents=True, exist_ok=True)
    predict_blocked(rf, scaler, IMG_RS, CLASSIFICATION_IMAGE, meta, nodata)
    logger.info(f"分类结果保存至: {CLASSIFICATION_IMAGE}")
    write_report(f"分类结果保存至: {CLASSIFICATION_IMAGE}")

    # 验证阶段
    logger.info("栅格化验证样本...")
    roi_v = rasterize_shapefile(VAL_SHP, IMG_RS, ATTRIBUTE)
    n_val = np.count_nonzero(roi_v)
    write_report(f"{n_val} 个验证像元")

    with rasterio.open(CLASSIFICATION_IMAGE) as pred_ds:
        pred_arr = pred_ds.read(1)
    X_v = pred_arr[roi_v > 0]
    y_v = roi_v[roi_v > 0]

    write_report("验证集混淆矩阵：")
    conv_mat = pd.crosstab(pd.Series(y_v, name='truth'), pd.Series(X_v, name='pred'), margins=True)
    write_report(conv_mat.to_string())

    oaa = accuracy_score(y_v, X_v)
    write_report(f"总体精度 (OAA) = {oaa * 100:.2f} %")

    cm_val = confusion_matrix(y_v, X_v)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm_val, annot=True, fmt='g')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('验证集混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.splitext(CLASSIFICATION_IMAGE)[0] + "_cm_val.png")
    plt.close()

    logger.info("分类任务完成。")

if __name__ == '__main__':
    main()

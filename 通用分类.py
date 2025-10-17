#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用遥感影像监督分类系统
-------------------------------------------------
功能：
1. 自动读取多波段遥感影像；
2. 从矢量样本中提取训练/验证数据；
3. 支持随机森林 / SVM / XGBoost 分类；
4. 采用分块预测模式；
5. 输出分类结果 GeoTIFF；
6. 自动生成分类报告与混淆矩阵。
"""

import os
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import mapping
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm

# ------------------ 参数配置 ------------------
IMAGE_PATH = r"D:\code313\Geo_programe\rasterio\RF\data\2017_09_05_stack.tif"
TRAIN_SHP = r"D:\code313\Geo_programe\rasterio\RF\data\cal.shp"
VAL_SHP = r"D:\code313\Geo_programe\rasterio\RF\data\val.shp"
ATTRIBUTE = "class"
OUT_DIR = Path("./results_v2")

CLASSIFIER = "rf"  # 可选: "rf", "svm", "xgb"
N_ESTIMATORS = 300
BLOCK_SIZE = 512
USE_GPU = False

OUT_DIR.mkdir(exist_ok=True)

# ------------------ 日志系统 ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "classification_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ 辅助函数 ------------------
def rasterize_samples(shp, ref_img, attr):
    """将矢量样本栅格化为与影像对齐的数组"""
    gdf = gpd.read_file(shp)
    gdf = gdf.to_crs(ref_img.rio.crs)
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attr]))
    mask = rxr.open_rasterio(ref_img.rio.path).rio.to_rasterio()
    import rasterio.features
    arr = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=ref_img.shape[1:],
        transform=ref_img.rio.transform(),
        fill=0,
        all_touched=True,
        dtype="uint16"
    )
    return arr


def extract_samples(image, mask):
    """根据掩膜提取样本特征与标签"""
    data = np.moveaxis(image.values, 0, -1)  # (bands, rows, cols) → (rows, cols, bands)
    valid = mask > 0
    X = data[valid]
    y = mask[valid]
    return X, y


def get_classifier(name):
    """构造分类器"""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS, n_jobs=-1, oob_score=True, verbose=1
        )
    elif name == "svm":
        return SVC(kernel="rbf", probability=True)
    elif name == "xgb":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=N_ESTIMATORS, learning_rate=0.1, max_depth=8, n_jobs=-1
            )
        except ImportError:
            raise ImportError("未安装 xgboost，请先运行 pip install xgboost")
    else:
        raise ValueError(f"未知分类器类型: {name}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title("混淆矩阵")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def predict_by_block(model, image, out_path, block_size=BLOCK_SIZE):
    """分块预测整幅影像"""
    height, width = image.shape[1], image.shape[2]
    profile = image.rio.profile
    profile.update(count=1, dtype="uint16", compress="lzw")

    with rxr.open_rasterio(image.rio.path) as src, \
            rxr.open_rasterio(out_path, mode="w", **profile) as dst:
        pass  # rioxarray 不直接支持写块，这里我们用 rasterio 写块
    import rasterio
    with rasterio.open(image.rio.path) as src, \
         rasterio.open(out_path, "w", **profile) as dst:
        for y in tqdm(range(0, height, block_size), desc="Block predicting"):
            h = min(block_size, height - y)
            window = rasterio.windows.Window(0, y, width, h)
            block = src.read(window=window)
            data = np.moveaxis(block, 0, -1).reshape(-1, block.shape[0])
            data = np.nan_to_num(data)
            preds = model.predict(data).reshape(h, width).astype("uint16")
            dst.write(preds, 1, window=window)
    return out_path


# ------------------ 主流程 ------------------
def main():
    t0 = time.time()
    logger.info("开始监督分类任务...")

    # 1. 读取影像
    img = rxr.open_rasterio(IMAGE_PATH, masked=True)
    logger.info(f"影像尺寸: {img.shape}, 波段数: {img.rio.count}")

    # 2. 训练样本栅格化与提取
    logger.info("正在处理训练样本...")
    train_mask = rasterize_samples(TRAIN_SHP, img, ATTRIBUTE)
    X_train, y_train = extract_samples(img, train_mask)
    logger.info(f"训练样本数: {len(y_train)}")

    # 3. 训练分类器
    clf = get_classifier(CLASSIFIER)
    logger.info(f"使用分类器: {clf.__class__.__name__}")
    clf.fit(X_train, y_train)
    logger.info("模型训练完成。")

    # 4. 精度评估（训练集）
    y_pred_train = clf.predict(X_train)
    acc = accuracy_score(y_train, y_pred_train)
    logger.info(f"训练集总体精度: {acc:.3f}")
    plot_confusion_matrix(y_train, y_pred_train, OUT_DIR / "train_cm.png")

    # 5. 分块预测整幅影像
    logger.info("开始分块预测...")
    classified_path = OUT_DIR / "classified_result.tif"
    predict_by_block(clf, img, classified_path)
    logger.info(f"分类结果保存至: {classified_path}")

    # 6. 验证阶段
    if os.path.exists(VAL_SHP):
        logger.info("正在进行验证...")
        val_mask = rasterize_samples(VAL_SHP, img, ATTRIBUTE)
        with rxr.open_rasterio(classified_path) as pred_img:
            pred_arr = pred_img.values.squeeze()
        Xv = pred_arr[val_mask > 0]
        yv = val_mask[val_mask > 0]
        acc_val = accuracy_score(yv, Xv)
        logger.info(f"验证总体精度: {acc_val:.3f}")
        plot_confusion_matrix(yv, Xv, OUT_DIR / "val_cm.png")

        report = classification_report(yv, Xv, digits=3)
        with open(OUT_DIR / "report.txt", "w", encoding="utf-8") as f:
            f.write(f"训练精度: {acc:.3f}\n验证精度: {acc_val:.3f}\n\n{report}")

    logger.info(f"全部任务完成，用时 {time.time()-t0:.1f} 秒。")


if __name__ == "__main__":
    main()

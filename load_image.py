import numpy as np
import os
from PIL import Image
import tensorflow as tf

'''
•load_image.py：載入單一影像與其對應標籤遮罩的工具，允許指定索引和子資料夾（Crosslines 或 Inlines），並將彩色標籤轉為數值遮罩。

檔案: load_image.py	
主要用途: 單張影像與標籤載入	
主要函式: get_image_and_label (根據索引讀取影像和對應彩色標籤，轉為數值遮罩)

'''
'''
load_image.py 檔案分析
此模組專門用於單張影像的讀取與標籤提取，通常在分析或視覺化時用到。

'''
def get_image_and_label(img_number, label, Crosslines=True):
    """
    Returns one RGB image and its corresponding label mask for a given index.
    
    Args:
        img_number (int): Index of the image to load (0-based).
        label (str): Name of the label subfolder (e.g., 'Line001').
        Crosslines (bool): If True, load from 'Crosslines', else 'Inlines'.
    
    Returns:
        tuple: (image_rgb_array, label_mask_array)
            - image_rgb_array: np.ndarray of shape (H, W, 3), dtype=uint8
            - label_mask_array: np.ndarray of shape (H, W), dtype=uint8
              Each pixel value is 0–5 depending on the class, or 255 if no match.
    """
    '''
    •參數：
        (1)img_number 為要讀取的影像索引（整數，從 0 開始），
        (2)label 為所屬標籤資料夾名稱（例如 'Line001'），
        (3)Crosslines 決定在資料路徑中使用「Crosslines」還是「Inlines」。
    •回傳：
        (1)回傳 (img_rgb, mask)。
        (2)其中 img_rgb 為形狀 (1, H, W, 3) 的 RGB 影像張量（經轉為 float32 /255.0 正規化），
        (3)mask 為形狀 (H, W) 的整數遮罩陣列，值在 0~5 代表對應類別，255 代表未匹配任何已定義顏色。
    •邏輯：
        (1)先根據 label 與 Crosslines 組出影像與標籤資料夾的絕對路徑（預設路徑為 /home/user/Desktop/TF_SEG2020/SEG2020/Images/<Label>/<Crosslines或Inlines> 和 /home/user/Desktop/TF_SEG2020/SEG2020/Images/Labels/<Crosslines或Inlines>）。
        (2)讀取該目錄下所有 .png 檔並排序。
        (3)檢查索引是否有效，取對應的影像檔與標籤檔路徑，用 PIL 開啟並轉 RGB，將影像轉為 NumPy uint8 陣列。
        (4)標籤影像則使用相同顏色轉換邏輯：函式內部定義六種 (R,G,B) 顏色對應六個類別。
        (5)將標籤影像展平後逐像素比對顏色，將符合者設為索引，不符合的設 255，最後重塑為 (H, W) 遮罩。
        (6)最後將原始影像加上批次維度並歸一化為 float32。此函式允許一次僅載入一張影像與其對應遮罩。
    '''
    # Define colors
    label_colors = np.array([
        (64, 67, 135),
        (34, 167, 132),
        (68, 1, 84),
        (41, 120, 142),
        (253, 231, 36),
        (121, 209, 81),
    ], dtype=np.uint8)

    # Paths
    subset = "Crosslines" if Crosslines else "Inlines"
    data_dir = os.path.join(f'/home/user/Desktop/TF_SEG2020/SEG2020/Images/{label}', subset)
    label_dir = os.path.join("/home/user/Desktop/TF_SEG2020/SEG2020/Images/Labels", subset)

    # Sorted file lists
    data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.png')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')])

    assert len(data_files) == len(label_files), f"Mismatch between data ({len(data_files)}) and label ({len(label_files)})"

    if img_number < 0 or img_number >= len(data_files):
        raise IndexError(f"img_number {img_number} is out of range (0–{len(data_files)-1}).")

    # Load image
    img_path = data_files[img_number]
    img_rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    # Load label and convert to mask
    lbl_path = label_files[img_number]
    lbl_rgb = np.array(Image.open(lbl_path).convert("RGB"), dtype=np.uint8)

    def pad_input(img):
        if img.ndim == 3:  # Image RGB
            return np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        elif img.ndim == 2:  # Label image
            return np.pad(img, ((1, 1), (1, 1)), mode='reflect')
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
    img_rgb = pad_input(img_rgb)
    lbl_rgb = pad_input(lbl_rgb)

    h, w, _ = lbl_rgb.shape
    label_flat = lbl_rgb.reshape(-1, 3)
    mask = np.full((label_flat.shape[0],), 255, dtype=np.uint8)  # default 255 for unmatched
    for idx, color in enumerate(label_colors):
        matches = np.all(label_flat == color, axis=1)
        mask[matches] = idx
    mask = mask.reshape(h, w)

    img_rgb = tf.expand_dims(img_rgb, axis=0)

    
    img_rgb = tf.cast(img_rgb, tf.float32)  / 255.0

    return img_rgb, mask

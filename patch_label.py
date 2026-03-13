import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
•patch_label.py：類似 patch.py，但以全圖中每個類別的區域為單位，逐一以指定顏色填充該類別區域，並評估分割表現。

檔案: patch_label.py	
主要用途: 類別遮罩實驗與評估	
主要函式: 
generate_images_by_class (對每個類別區域填色生成圖片和遮罩)、
mean_pixel_value、
prediction_with_mask、
evaluate_predictions、evaluate_original

'''
'''
patch_label.py 檔案分析
patch_label.py 與 patch.py 類似，差別在於遮罩方式：此處以 類別區域 為單位進行覆蓋。可用於評估將整個一類別（如某區塊）替換為指定顏色後的模型預測改變。核心函式如下：
'''

def generate_images_by_class(img: np.ndarray, filler: tuple[float, float, float], label: np.ndarray, num_classes: int = 6):
    """
    Génère une liste d'images où chaque classe est remplacée par la couleur filler,
    ainsi que les masques binaires correspondants.
    
    Args:
        img (np.ndarray): image d'entrée (H, W, 3) ou (1, H, W, 3)
        filler (tuple): couleur de remplacement (R, G, B)
        label (np.ndarray): masque de classes (H, W) avec valeurs entières entre 0 et num_classes-1
        num_classes (int): nombre de classes
    
    Returns:
        images (list[np.ndarray]): liste de num_classes images (1, H, W, 3)
        masks (list[np.ndarray]): liste de num_classes masques binaires (H, W)
    """
    '''
    •參數：
        (1)img 為輸入 RGB 影像（(H,W,3) 或 (1,H,W,3)），
        (2)filler 為填充顏色，
        (3)label 為真實遮罩（(H,W)，值為 0~num_classes-1），
        (4)num_classes 類別數。
    •回傳：
        回傳 (images, masks)，
        - 其中 images 為長度 
        - num_classes 的列表，每張圖形狀 (1,H,W,3)，該張圖對應類別 c，將原圖中 label==c 的像素替換為 filler 顏色；
        - masks 為相對應的 num_classes 個布林遮罩（(H,W)），每個遮罩指示該圖中被替換的像素位置（即 label==c）。
    •邏輯：
        (1)去除可能的批次維度後取得 (H,W,3)。
        (2)對於每個類別 c，先生成布林遮罩 mask = (label == c)，然後複製原圖並對所有 mask==True 的位置賦值為 filler。
        (3)加回批次維度後將此影像加入 images，同時將布林遮罩加入 masks。
        (4)重複 0 到 num_classes-1。
        (5)最終返回填色後的影像列表及對應遮罩列表。
    '''
    # enlever le batch channel si nécessaire
    if img.ndim == 4:
        img = img[0]
    H, W, _ = img.shape
    
    images = []
    masks = []
    
    for c in range(num_classes):
        # masque booléen de la classe
        mask = (label == c)
        
        # copie de l'image originale
        new_img = img.numpy().copy()
        # appliquer le filler
        new_img[mask] = filler
        # rajouter la dimension batch
        new_img = np.expand_dims(new_img, axis=0)
        
        images.append(new_img)
        masks.append(mask)
    
    return images, masks


# fonction calculant la moyeen des pixels par chanel d4un image:
def mean_pixel_value(image: np.ndarray):
    """
    Calcule la moyenne des valeurs des pixels par canal d'une image.
    Args:
        image (np.ndarray): image d'entrée (H, W, 3)
    Returns:
        tuple: moyenne des valeurs des pixels par canal (R, G, B)
    """
    '''
    此函式與 patch.py 中同名函式完全相同，計算影像在每個通道的平均值。
    '''
    if image.ndim == 4:
        image = image[0]  # enlever le channel batch
    return tuple(np.mean(image, axis=(0, 1)))  # moyenne sur les dimensions H et W




def evaluate_predictions(results, list_mask, pred_mask, num_classes=6):
    """Compute mean accuracy and mean IoU for predictions, ignoring masked regions.
    Args:
        results (list of np.ndarray): list of predictions, each (1,H,W,C)
        list_mask (list of tuple): list of pixel corresponding to the masked area
        pred_mask (np.ndarray): ground truth (H,W), values in [0..num_classes-1]
        num_classes (int): number of classes 
    Returns:
        dict: {"mean_accuracy": float, "mean_mIoU": float}"""
    '''
    •參數：
        (1)results 為遮罩後影像的預測結果列表（每項形狀 (1,H,W,C)），
        (2)list_mask 為布林遮罩列表（每項形狀 (H,W)），
        (3)pred_mask 為原始真實遮罩，
        (4)num_classes 類別數。
    •回傳：
        同 patch.py，回傳 (metric_name, value) 列表。
    •邏輯：
        (1)與 patch.py 的 evaluate_predictions 類似，只是這裡傳入的是布林遮罩（指示整個類別區域）而非方形座標。
        (2)對每個預測結果，去批次維度取 (H,W,C)，argmax 得標籤。
        (3)對於遮罩真值，使用布林遮罩過濾掉被替換的區域，僅對未被替換的像素計算準確度與 IoU。
        (4)同樣收集各類別 IoU，計算總體 mIoU，以及各類別平均 IoU，最後輸出。
    '''
    H, W = pred_mask.shape
    acc_list = []
    miou_list = []
    class_ious = {cls: [] for cls in range(num_classes)}  # store IoUs per class across images
    for pred, mask in zip(results, list_mask):
        # remove batch dim
        pred = pred[0]  # (H,W,C)
        # predicted labels
        pred_labels = np.argmax(pred, axis=-1)  # (H,W)
        valid = ~mask
        print(f"valid shape: {valid.shape}, pred_mask shape: {pred_mask.shape}, pred_labels shape: {pred_labels.shape}")
        y_true = pred_mask[valid]
        y_pred = pred_labels[valid]
        # -------- Accuracy --------
        acc = np.mean(y_true == y_pred)
        acc_list.append(acc)
        # -------- IoU per class --------
        ious = []
        for cls in range(num_classes):
            true_cls = (y_true == cls)
            pred_cls = (y_pred == cls)
            intersection = np.logical_and(true_cls, pred_cls).sum()
            union = np.logical_or(true_cls, pred_cls).sum()
            if union > 0:
                iou = intersection / union
                ious.append(iou)
                class_ious[cls].append(iou)
        if len(ious) > 0:
            miou_list.append(np.mean(ious))
    # compute mean IoU per class across all images
    mean_iou_per_class = {
        cls: float(np.mean(vals)) if len(vals) > 0 else None
        for cls, vals in class_ious.items()}
    dic_results= {        "mean_accuracy": float(np.mean(acc_list)),
                          "mean_mIoU": float(np.mean(miou_list)),
                          "mean_IoU_per_class": mean_iou_per_class}
    result_list = []
    for key, value in dic_results.items():
        if isinstance(value, dict):
            # If it's a dict (like mean_IoU_per_class), add each sub-item
            for subkey, subval in value.items():
                result_list.append((f"{key}_{subkey}", subval))
        else:
            result_list.append((key, value))
    return result_list



def prediction_with_mask(images_with_filler, model): 
    '''
    功能與 patch.py 中相同：
    將傳入的遮罩後影像列表轉為張量並逐張使用模型預測，返回結果列表。
    '''
    for i in range(len(images_with_filler)):
        images_with_filler[i] = tf.convert_to_tensor(images_with_filler[i], dtype=tf.float32)
    # Analyse des images avec le modèle
    results = []
    for img in images_with_filler:
        result = model.predict(img)
        results.append(result)
    return results







def evaluate_original(image, label, model, num_classes=6):
    """Evaluate model on the original (unmasked) image.
    Args:
        image (np.ndarray): input image, shape (1,H,W,3)
        label (np.ndarray): ground truth labels, shape (H,W)
        model: trained model with .predict()
        num_classes (int): number of classes
    Returns:
        dict: {"mean_accuracy": float,
               "mean_mIoU": float,
               "mean_IoU_per_class": dict}"""
    '''
    功能與 patch.py 中相同：對完整原圖計算準確度與類別 IoU，回傳結果列表。
    '''
    # predict
    pred = model.predict(image)   # (1,H,W,C)
    pred = pred[0]                # remove batch dim
    pred_labels = np.argmax(pred, axis=-1)  # (H,W)
    # -------- Accuracy --------
    acc = np.mean(label == pred_labels)
    # -------- IoU per class --------
    ious = []
    class_ious = {}
    for cls in range(num_classes):
        true_cls = (label == cls)
        pred_cls = (pred_labels == cls)

        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            class_ious[cls] = float(iou)
        else:
            class_ious[cls] = None  # class not present
    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    dic_results = {        "mean_accuracy": float(acc),
                            "mean_mIoU": miou,
                            "mean_IoU_per_class": class_ious}
    result_list = []
    for key, value in dic_results.items():
        if isinstance(value, dict):
            # If it's a dict (like mean_IoU_per_class), add each sub-item
            for subkey, subval in value.items():
                result_list.append((f"{key}_{subkey}", subval))
        else:
            result_list.append((key, value))
    return result_list
    

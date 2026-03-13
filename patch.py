import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import tensorflow as tf

'''
•patch.py：針對影像進行遮罩實驗的工具，主要功能為在原圖上以指定顏色填充固定大小的方形區塊，並計算覆蓋區域對分割預測準確度與 IoU 的影響。

檔案: patch.py	
主要用途: 影像遮罩實驗與評估	
主要函式: 
mean_pixel_value (計算平均色值)、
generate_images_with_filler (移動方塊遮罩生成圖片)、
prediction_with_mask (模型推論遮罩影像)、
evaluate_predictions (計算遮罩區域忽略後的準確度與 IoU)、
evaluate_original (原圖評估)
'''
'''
patch.py 檔案分析
patch.py 提供對影像進行方塊遮罩（patch-based occlusion）分析的功能，用以評估遮罩對模型分割結果的影響。
主要包含生成含遮罩的影像、用模型推論，以及計算「忽略遮罩區域」後的評估指標（準確率、IoU）等。
'''


def mean_pixel_value(image: np.ndarray):
    """
    Calcule la moyenne des valeurs des pixels par canal d'une image.
    Args:
        image (np.ndarray): image d'entrée (H, W, 3)
    Returns:
        tuple: moyenne des valeurs des pixels par canal (R, G, B)
    """
    '''
    •參數：
        image 為輸入影像陣列，形狀可為 (H, W, 3) 或包含批次維度 (1, H, W, 3)。
    •回傳：
        回傳 (mean_R, mean_G, mean_B) 三元組，為影像在各通道的平均像素值。
    •邏輯：
        (1)若影像為 4D（第0維為 batch），先去掉批次維度，取得 (H, W, 3)。
        (2)然後對空間維度做平均，計算每個通道的平均值，並以 tuple 形式返回。
        (3)此函式可用來取得某張影像的全局平均色值。
    '''
    if image.ndim == 4:
        image = image[0]  # enlever le channel batch
    return tuple(np.mean(image, axis=(0, 1)))  # moyenne sur les dimensions H et W




def generate_images_with_filler(img: np.ndarray, filler: tuple[float, float, float], l_gap: int, L_gap: int):
    """
    Génère une liste d'images à partir d'une image RGB en remplaçant
    un carré de taille l_gap × l_gap par une couleur filler.
    Args:
        img (np.ndarray): image d'entrée (H, W, 3)
        filler (tuple): triplet de flottants (R, G, B)
        l_gap (int): taille du carré à remplir
        L_gap (int): pas de déplacement du carré
    Returns:
        list[np.ndarray]: liste des nouvelles images
    """
    '''
    •參數：
        (1)img 為輸入 RGB 影像（或形狀 (1,H,W,3) 的張量），
        (2)filler 為三元組顏色值，用於填充遮罩區域（例如黑色或灰色），
        (3)l_gap 為遮罩方形區域邊長（正方形大小），
        (4)L_gap 為遮罩移動的步長。
    •回傳：
        回傳二元組 (images, mask_list)。
            - images 為列表，內含若干張與原圖相似但在不同位置有矩形區域被填充 filler 顏色的影像，每張圖保持形狀 (1, H, W, 3)；
            - mask_list 為列表，對應每張圖中被填充區域的座標 (y1, x1, y2, x2)。
    •邏輯：
        (1)首先去掉批次維度得到 (H, W, 3)。
        (2)使用雙層迴圈，以 L_gap 的步長在影像高度與寬度上遍歷，對每個位置從 (y, x) 開始，將大小為 l_gap×l_gap 的子區域填充為 filler 顏色。
        (3)為避免修改原圖，每次都複製原圖（img.numpy().copy()），對副本塗上顏色，然後再加回批次維度展成 (1,H,W,3)。
        (4)將此遮罩後的影像加到 images 列表中，並記錄該遮罩方塊的位置座標到 mask_list。
        (5)最終返回所有生成的遮罩影像和對應座標清單。此函式可用於生成一系列「挖洞遮罩」影像，觀察模型在不同遮罩位置的反應。
    '''
    #enelever le channel batch de l'image
    if img.ndim == 4:
        img = img[0]
    H, W, _ = img.shape
    images = []
    mask_list=[]
    for y in range(0, H - l_gap + 1, L_gap):
        for x in range(0, W - l_gap + 1, L_gap):
            # copie de l'image originale
            new_img = img.numpy().copy()
            # remplissage du carré
            new_img[y:y+l_gap, x:x+l_gap, :] = filler
            #ajout du chanel batch
            new_img = np.expand_dims(new_img, axis=0)
            # ajout de l'image à la liste
            images.append(new_img)
            coordinates = (y, x, y + l_gap, x + l_gap)
            mask_list.append(coordinates)
    return images,mask_list

def prediction_with_mask(images_with_filler, model): 
    '''
    •參數：
        (1)images_with_filler 為上述生成的遮罩影像列表，內含多個形如 (1,H,W,3) 的張量，
        (2)model 為訓練好的模型。
    •回傳：
        回傳 results 列表，每個元素為對應輸入影像的模型預測結果（通常形狀 (1, H, W, C) 的 NumPy 陣列）。
    •邏輯：
        (1)首先將每張影像轉為 TensorFlow 張量並設定 dtype=tf.float32。
        (2)然後對列表中的每張影像呼叫 model.predict(img) 取得預測，並將結果存入列表 results。
        (3)註：這裡逐張預測而非一次批次運算。最後返回所有預測結果。
    '''
    for i in range(len(images_with_filler)):
        images_with_filler[i] = tf.convert_to_tensor(images_with_filler[i], dtype=tf.float32)
    # Analyse des images avec le modèle
    results = []
    for img in images_with_filler:
        result = model.predict(img)
        results.append(result)
    return results





def evaluate_predictions(results, list_mask, pred_mask, num_classes=6):
    """Compute mean accuracy and mean IoU for predictions, ignoring masked regions.
    Args:
        results (list of np.ndarray): list of predictions, each (1,H,W,C)
        list_mask (list of tuple): list of (y1,x1,y2,x2) coords for the filler region
        pred_mask (np.ndarray): ground truth (H,W), values in [0..num_classes-1]
        num_classes (int): number of classes 
    Returns:
        dict: {"mean_accuracy": float, "mean_mIoU": float}"""
    '''
    •參數：
        (1)results 為 prediction_with_mask 的輸出列表，每個元素形狀為 (1, H, W, C) 的預測張量；
        (2)list_mask 為對應的遮罩座標列表，每個元素為 (y1, x1, y2, x2)；
        (3)pred_mask 為原始影像的真實單通道遮罩（形狀 (H, W)），值為 0~num_classes-1 類別；
        (4)num_classes 為類別數（預設 6）。
    •回傳：
        回傳列表 result_list，每項為 (metric_name, value) 的元組，
            例如:
            - ("mean_accuracy", 0.92)、
            - ("mean_mIoU", 0.80) 以及 
            - ("mean_IoU_per_class_0", 0.75)、
            - ("mean_IoU_per_class_1", 0.88) 等。
    •邏輯：
        (1)對每一張遮罩影像的預測結果與其遮罩座標進行評估：
        (2)首先去掉預測結果的批次維度得到 (H,W,C) 的概率，並取最大值作為預測標籤 (H,W)。
        (3)然後根據遮罩座標 (y1,x1,y2,x2) 建立一個布林遮罩，標記遮罩區域內為 True（忽略區域）。
        (4)接著只對非遮罩區域（mask==False）的像素計算指標：計算分類準確率 acc = mean(y_true == y_pred)，並對每一類別計算 IoU 值（交集/聯集）存入列表。
        (5)若某類別在有效區域中出現聯集，則累積該類 IoU。對一張圖，平均所有出現類別的 IoU 作為當前遮罩圖的 mIoU，加入列表。
        (6)重複所有遮罩圖後，將所有遮罩圖的準確率取平均作為 mean_accuracy，所有遮罩圖的 mIoU 取平均作為 mean_mIoU。
        (7)另外計算每個類別在所有遮罩圖中的平均 IoU，輸出為 mean_IoU_per_class_x。最後將這些指標以列表方式返回。此函式衡量模型在「忽略指定方塊區域」的情況下預測其餘區域的品質。
    '''
    H, W = pred_mask.shape
    acc_list = []
    miou_list = []
    class_ious = {cls: [] for cls in range(num_classes)}  # store IoUs per class across images
    for pred, coords in zip(results, list_mask):
        # remove batch dim
        pred = pred[0]  # (H,W,C)
        # predicted labels
        pred_labels = np.argmax(pred, axis=-1)  # (H,W)
        # build mask
        y1, x1, y2, x2 = coords
        mask = np.zeros((H, W), dtype=bool)
        mask[y1:y2, x1:x2] = True  # True = ignore region
        # apply mask
        valid = ~mask
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
    •參數：
        (1)image 為完整影像（形狀 (1, H, W, 3)），
        (2)label 為完整真實遮罩（形狀 (H, W)），
        (3)model 為訓練模型，num_classes 類別數。
    •回傳：
        回傳 (metric_name, value) 列表，
        例如
        - ("mean_accuracy", 0.95)、
        - ("mean_mIoU", 0.85)、
        - ("mean_IoU_per_class_0", 0.80), ...。
    •邏輯：
        (1)對未遮罩原圖進行評估：
        (2)先用 model.predict(image) 得到預測（去除 batch 維度後 (H,W,C)），取 argmax 生成預測標籤 (H,W)。
        (3)計算影像全域準確率 acc，然後對每個類別計算 IoU，將存在的類別的 IoU 存入 ious 列表並更新 class_ious[cls]，未出現則設定為 None。計算此圖的 mIoU（對出現的類別求均值）。
        (4)然後構建結果字典包含 mean_accuracy, mean_mIoU, 和每類別的 IoU。
        (5)最後把字典拆分為 (key, value) 格式的列表後回傳。此結果可以作為原始影像分割性能的基準。
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
    


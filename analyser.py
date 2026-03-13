import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage.transform import resize

'''
•analyser.py：提供針對訓練完成之分割模型的可視化與分析工具，包括生成 Grad-CAM 熱度圖、信心度地圖、顯示特徵圖、加權特徵圖等功能。

檔案: analyser.py	
主要用途: 分割模型的 Grad-CAM 與激活圖分析	
主要函式: 
make_gradcam_heatmap (生成 Grad-CAM)、
visualize_gradcam_all_classes (繪製各類別 Grad-CAM)、
get_confidence_map (信心度圖)、
visualize_confidence_percentiles (依百分位繪圖)、
show_activation_maps (顯示特徵圖)、
show_weighted_maps (繪製加權特徵圖)、
get_weighted_activation_maps (計算加權特徵圖)

'''
'''
analyser.py 檔案分析
此模組針對已訓練的分割模型提供視覺化分析工具。主要功能包括生成 Grad-CAM 熱度圖、在多類別分割情境下逐類別和合併顯示熱度圖、計算每像素的分類信心度，以及顯示中間特徵圖等。
'''
########################################################################################### HEAT MAP #######################################################################################################################

def make_gradcam_heatmap(img_array, model, conv_layer_name, pred_index=None):
    '''
    •參數：
        (1)img_array 為待分析之影像張量（形狀通常為 (1, H, W, 3)）
        (2)model 為訓練好的 tf.keras.Model
        (3)conv_layer_name 指定 Grad-CAM 所用的卷積層名稱
        (4)pred_index（可選）指定要分析的類別索引（預設取最大預測分數之類別）。
    •回傳：回傳一個 2D 熱度圖陣列（形狀 (H', W')，對應特徵圖尺寸）。
    •邏輯：此函式先建立一個「子模型」，輸入為原始模型的輸入，輸出為目標卷積層的輸出與最終預測。
        然後使用 TensorFlow 的 GradientTape 計算預測得分對於卷積層輸出的梯度。
        對梯度在空間維度做全域平均（求每通道的重要性權重），再將原始卷積輸出與權重相乘並對通道求和，產生原始的 CAM 地圖。
        最後對結果取 ReLU，並根據第10與第99百分位做剪裁與對比度增強，使結果介於 0 到 1 之間，作為熱度圖回傳。
    '''
    # Create a model that maps the input image to the activations of the conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0], axis=-1)
        # If pred_index is an int, make it a tensor
        if isinstance(pred_index, int):
            pred_index = tf.constant(pred_index)
        # Gather the score for the class
        class_channel = predictions[..., pred_index]

    # This is the gradient of the output neuron (top predicted or chosen class) with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool the gradients over all the axes except for the channel
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # Multiply each channel in the feature map array by "how important this channel is" with regard to the class
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap_np = heatmap.numpy()
    heatmap = tf.nn.relu(heatmap)                                                # Contributions positives
    
    # Clipping enhance contrast
    low = np.percentile(heatmap, 10)  # lower 10%
    high = np.percentile(heatmap, 99)  # upper 1%
    if high - low > 1e-6:
        heatmap = np.clip((heatmap - low) / (high - low), 0, 1)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap

def visualize_gradcam_all_classes(model, sample_image, make_gradcam_heatmap_fn, preds, conv2d_index_from_end=2):
    """
    Generates and visualizes Grad-CAM heatmaps for all classes in a segmentation model.

    Args:
        model (tf.keras.Model): The trained model.
        sample_image (np.ndarray): A single RGB image of shape (H, W, 3).
        make_gradcam_heatmap_fn (callable): Function to compute Grad-CAM heatmap.
        preds (np.ndarray): Prediction output from the model, shape (1, H, W, num_classes).
        conv2d_index_from_end (int): Index of the Conv2D layer from the end to use for Grad-CAM (default is 2 = before final 1x1 conv).

    Returns:
        None. Displays matplotlib plots.
    """
    '''
    •參數：
        (1)model 為分割模型
        (2)sample_image 為單張 RGB 影像（形狀 (H, W, 3)）
        (3)make_gradcam_heatmap_fn 為用以上函式產生熱度圖的可呼叫物件，preds 為模型對 sample_image 的預測輸出（形狀 (1, H, W, C)）
        (4)conv2d_index_from_end 指定倒數第幾個 Conv2D 層用於 Grad-CAM（預設第2個）
    •回傳：
        (1)本函式主要功能是繪製圖表（無實際回傳值，若成功則顯示多張熱度圖）。
        (2)其最後步驟回傳合併熱度圖（形狀與 heatmap 相同）。
    •邏輯：
        (1)先從模型的所有層中逆序尋找指定順序的卷積層名稱。
        (2)接著將輸入影像加維度變為 (1, H, W, 3)，
        (3)找出模型最後一層輸出通道數 num_classes（即類別數）。
        (4)對每個類別通道 ch，呼叫 make_gradcam_heatmap_fn(img, model, last_conv_layer_name, pred_index=ch) 取得該類別的熱度圖；
        (5)然後顯示所有類別的熱度圖，並以「每像素取所有類別熱度最大值」為準則合併生成「Combined」熱度圖，
        (6)再將原始影像與該熱度圖以半透明方式疊加顯示。整體以 matplotlib 繪圖顯示各圖（包括每類別熱度圖與合併圖），同時在控制台顯示所用層名稱資訊。
    '''
    # Step 1: Find the N-th last Conv2D layer
    conv2d_count = 0
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv2d_count += 1
            if conv2d_count == conv2d_index_from_end:
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("Could not find a Conv2D layer at the specified index.")

    print(f"[INFO] Using layer for Grad-CAM: {last_conv_layer_name}")

    # Step 2: Prepare the input image
    img = sample_image[np.newaxis, ...]  # shape (1, H, W, 3)

    # Step 3: Generate Grad-CAM heatmaps for each class
    #num_classes = preds.shape[-1]
    num_classes = model.output_shape[-1]

    heatmaps = []
    for ch in range(num_classes):
        heatmap = make_gradcam_heatmap_fn(img, model, last_conv_layer_name, pred_index=ch)
        heatmaps.append(heatmap)

    # Step 4: Plot heatmaps per class
    plt.figure(figsize=(3 * num_classes, 3))
    for ch in range(num_classes):
        plt.subplot(1, num_classes, ch + 1)
        plt.imshow(heatmaps[ch], cmap='hot')
        plt.title(f"Grad-CAM: Class {ch}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Step 5: Combined heatmap (max over channels)
    combined_heatmap = np.max(np.stack(heatmaps, axis=-1), axis=-1)

    plt.figure(figsize=(6, 6))
    plt.imshow(sample_image)
    plt.imshow(combined_heatmap, cmap='hot', alpha=0.5)
    plt.title("Grad-CAM: Combined (max over classes)")
    plt.axis('off')
    plt.show()
    return combined_heatmap

############################################################################################# CONFIDENCE ###########################################################################################################################
def get_confidence_map(img_array, model):
    """
    Computes per-pixel confidence as the max softmax prob at each pixel.
    img_array: shape (1, H, W, 3)
    Returns: (H, W) confidence map
    """
    '''
    •參數：
        (1)img_array（形狀 (H, W, 3) 或 (1, H, W, 3) 的單張影像）
        (2)model 為分割模型
    •回傳：回傳一個 (H, W) 的信心度地圖，數值介於 0 到 1 之間。
    •邏輯：
        (1)先將 img_array 加入 batch 維度後帶入模型計算預測（輸出形狀 (1, H, W, C)）。
        (2)對最後一個維度（類別維）做 softmax，得到每像素各類別的機率，
        (3)然後取每個像素的最大機率值，作為該像素的分類信心度。
        (4)最後除去 batch 維度，回傳純 2D 陣列。
    '''
    predictions = model(np.expand_dims(img_array, axis=0))  # shape: (1, H, W, C)
    softmax_probs = tf.nn.softmax(predictions, axis=-1)
    confidence_map = tf.reduce_max(softmax_probs, axis=-1)  # shape: (1, H, W)
    return confidence_map[0].numpy()

def visualize_confidence_percentiles(conf_map, percentiles=[0, 10, 20, 50, 60, 99], cmap='viridis'):
    """
    Visualizes confidence maps by thresholding at different top-% levels.

    Args:
        conf_map (np.ndarray): A 2D confidence or heatmap array.
        percentiles (list of int): Percentiles to visualize (e.g. [0, 10, 20, 50]).
        cmap (str): Matplotlib colormap to use for plotting.

    Returns:
        None. Displays matplotlib plots.
    """
    '''
    •參數：
        (1)conf_map 為信心度地圖（2D 陣列）
        (2)percentiles 為要顯示的百分位列表
        (3)cmap 為 matplotlib 的色彩映射方式。
    •回傳：本函式僅產生可視化圖表（無明確回傳值）。
    •邏輯：
        (1)將信心度地圖展平成長向量，排序後依據指定百分位計算閾值。
        (2)對於每個百分位 p，計算「Top (100-p)% 的閾值」，將高於閾值的像素以原信心度顯示，低於者設為 0。
        (3)然後對每個閾值生成子圖顯示信心度遮罩圖（使用指定色彩映射），並在右側附上顏色條。
        (4)此功能可協助視覺化不同信心水準下的像素分佈。
    '''
    n = len(percentiles)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    flat_conf = conf_map.flatten()
    sorted_vals = np.sort(flat_conf[flat_conf > 0])  # exclude zero pixels
    total_pixels = len(sorted_vals)

    for i, p in enumerate(percentiles):
        k = int((100 - p) / 100 * total_pixels)
        if k == 0:
            threshold = sorted_vals[-1] + 1e-6
        else:
            threshold = sorted_vals[-k]

        confident_pixels = np.where(conf_map >= threshold, conf_map, 0)

        ax = axes[i]
        im = ax.imshow(confident_pixels, cmap=cmap, vmin=threshold, vmax=conf_map.max())
        ax.set_title(f"Top {100 - p}%\n(≥ {threshold:.4f})")
        ax.axis('off')

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.show()

##################################################################################### ACTIVATION MAP ####################################################################################################################################


def show_activation_maps(model, image, layer_name, crop_percent=0.05):
    """
    Displays all activation maps (feature maps) of a specific layer for a given image,
    with top cropping based on strongest activations.

    Args:
        model (tf.keras.Model): Your model.
        image (np.ndarray): Input image of shape (H, W, 3).
        layer_name (str): Name of the layer to extract activations from.
        crop_percent (float): Fraction of top activations to base cropping on (default 0.05).

    Returns:
        activations: np.ndarray of shape (H', W', C)
    """
    '''
    •參數：
        (1)model 為分割模型
        (2)image 為輸入影像（形狀 (H, W, 3)）
        (3)layer_name 為要取出的層名稱
        (4)crop_percent（預設 0.05）指定在繪圖時裁切多少比例以保留最強激活。
    •回傳：
        (1)回傳該層的激活圖陣列 activations（形狀 (H', W', C)，其中 C 為該卷積層的通道數）。
    •邏輯：
        (1)首先將輸入影像保證為 4 維 (加 batch)。
        (2)建立一個 activation_model，
        (3)輸出為指定層的激活圖。
        (4)執行前向傳播得到 activations（形狀 (1, H', W', C)），移除 batch 維度得 (H', W', C)。
        (5)程式會對每個通道的激活圖標準化到 [0,1]，然後根據 crop_percent 設定找到前 crop_percent 的活動熱點行位置，對每個通道圖從該行往下裁切，再將裁切後的區域縮放回原始高度 (H')。
        (6)最後以 Jet 色彩映射並置於網格中顯示所有通道特徵圖，方便觀察各濾波器在該層對輸入的響應。
    '''
    image = np.expand_dims(image, axis=0) if image.ndim == 3 else image

    activation_model = tf.keras.Model(inputs=model.inputs,
                                      outputs=model.get_layer(layer_name).output)

    activations = activation_model.predict(image)[0]  # shape: (H, W, C)
    H, W, C = activations.shape

    n_cols = 8
    n_rows = int(np.ceil(C / n_cols))
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    for i in range(C):
        channel = activations[..., i]

        # Normalize
        ch_min = np.min(channel)
        ch_max = np.max(channel)
        if ch_max - ch_min > 1e-5:
            norm_channel = (channel - ch_min) / (ch_max - ch_min)
        else:
            norm_channel = np.zeros_like(channel)

        # Flatten and get threshold for top crop_percent activations
        flat = norm_channel.flatten()
        threshold = np.quantile(flat, 1.0 - crop_percent)

        # Find the first row index (from top) where this threshold is crossed
        mask = norm_channel >= threshold
        rows_with_activation = np.any(mask, axis=1)
        first_active_row = np.argmax(rows_with_activation)

        # Crop from that row downward
        cropped = norm_channel[first_active_row:, :]

        # Resize back to original height for visualization consistency
        resized = resize(cropped, (H, W), preserve_range=True, anti_aliasing=True)

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(resized, cmap='jet')
        plt.axis('off')
        plt.title(f"Ch {i}")

    plt.tight_layout()
    plt.show()

    return activations


def show_weighted_maps(weighted_maps):
    '''
    •參數：
        (1)weighted_maps 為加權特徵圖陣列（形狀 (H', W', C)），通常由 get_weighted_activation_maps 產生。
    •回傳：
        無回傳值，此函式會繪製圖表。
    •邏輯：
        (1)對於每個通道（共 C 個），找出該通道加權激活圖中的最大值位置，
        (2)然後計算前 N 個（預設取 8 個）最大值之間的間距比例作為直線寬度參數。
        (3)將所有通道的加權激活圖與直線框疊加後，以 Jet 色彩映射顯示。
        (4)此可視化用於檢視每個通道的重要激活區域。
    '''
    num_channels = weighted_maps.shape[-1]
    n_cols = 8
    n_rows = int(np.ceil(num_channels / n_cols))
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for i in range(num_channels):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(weighted_maps[..., i], cmap='jet')
        plt.axis('off')
        plt.title(f"Weighted Ch {i}")
    plt.tight_layout()
    plt.show()


    
def get_weighted_activation_maps(model, image, layer_name, class_index):
    """
    Returns each activation map of the given layer, weighted by its gradient w.r.t. the class output.

    Args:
        model (tf.keras.Model): Your model.
        image (np.ndarray): Input image of shape (H, W, 3).
        layer_name (str): Layer to extract activations from.
        class_index (int): Output class to target.

    Returns:
        weighted_maps: np.ndarray of shape (H', W', C) – each channel weighted by gradient.
        raw_activations: np.ndarray of shape (H', W', C)
    """
    '''
    •參數：
        (1)model 為分割模型
        (2)image 為輸入影像（形狀 (H, W, 3)）
        (3)layer_name 為卷積層名稱，class_index 為目標類別索引。
    •回傳：
        (1)回傳二元元組 (weighted_maps, raw_activations)。
        (2)兩者皆為形狀 (H', W', C) 的浮點陣列：weighted_maps 是對應每個通道乘上平均梯度權重後的加權激活圖；
        (3)raw_activations 是原始未加權的激活圖。
    •邏輯：
        (1)本質同 Grad-CAM 原理。
        (2)先擴增影像為 (1,H,W,3)，建立子模型輸出指定卷積層與最終預測，使用 GradientTape 計算目標類別分數對卷積輸出的梯度。
        (3)將梯度在空間維度上平均得到每通道權重，然後將原始卷積輸出（不含 batch 維度）逐通道乘以該權重，得到每通道的加權激活圖矩陣。
        (4)返回加權圖與原始激活圖供後續處理或顯示。
    '''
    image = np.expand_dims(image, axis=0)

    # Build gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        # Assuming segmentation output: shape (1, H, W, num_classes)
        class_scores = tf.reduce_mean(predictions[..., class_index])  # Global class activation

    # Compute gradients of class score w.r.t. conv outputs
    grads = tape.gradient(class_scores, conv_outputs)[0]  # (H', W', C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))     # (C,)

    conv_outputs = conv_outputs[0]  # (H', W', C)

    weighted_maps = conv_outputs * pooled_grads  # broadcast (H', W', C) * (C,)

    return weighted_maps.numpy(), conv_outputs.numpy()


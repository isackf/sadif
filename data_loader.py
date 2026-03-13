import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import tensorflow as tf

'''
•data_loader.py：負責從磁碟讀取影像與標籤資料，將 RGB 標籤影像轉換為單通道類別遮罩，並將資料集分割為訓練、驗證、測試集，最後封裝為 TensorFlow 的 tf.data.Dataset。

檔案:data_loader.py	
主要用途:資料讀取與預處理	
主要函式: 
split_data (載入影像資料、轉換標籤、分割訓/驗/測集)、
easy_load_data (便利載入指定子資料夾)、
load_tf_dataset (建立 tf.data.Dataset)；
內部還定義了 load_images_RGB, convert_label_to_1ch, pad_input, convert_to_tf_dataset 等輔助函式


'''
'''
data_loader.py 檔案分析
此模組負責從檔案系統載入影像資料，將彩色標籤轉換為數值遮罩，並將資料集分割與批次化。

'''

def split_data(data_folder_path,
               label_folder_path,
               batch_size,
               Crosslines=True,
               percentage_training=0.7,
               percentage_test=0.2):
    """
    Charge les images, transforme les labels RGB en masques 6 canaux (one-hot spatial pour 6 couleurs),
    puis split en train / val / test et les batch.

    Les 6 couleurs de label attendues sont :
        [(64, 67, 135), (34, 167, 132), (68, 1, 84),
         (41, 120, 142), (253, 231, 36), (121, 209, 81)]
    """

    '''
    •參數：
        (1)data_folder_path 為儲存影像的資料夾路徑，
        (2)label_folder_path 為對應標籤的資料夾路徑，
        (3)batch_size 為每個批次大小，
        (4)Crosslines 選擇使用的資料子集（若 True，在子目錄「Crosslines」下找資料，否則在「Inlines」），
        (5)percentage_training/percentage_test 分別為訓練集與測試集佔比（剩餘自動為驗證集）。
    •回傳：
        (1)回傳三個列表 train_dataset, test_dataset, val_dataset，均為 tf.data.Dataset 類型，
        (2)元素為 (影像張量, 標籤遮罩)。
    •邏輯：
        (1)首先列出 data_folder_path/<label>/<Crosslines或Inlines> 下所有 .png 檔案（彩色影像）和 label_folder_path/<Crosslines或Inlines> 下所有標籤 .png。
        (2)確認影像數量與標籤數量相同。然後定義兩個內部函式：
    •load_images_RGB(file_list)：
        (1)讀取每個影像檔（PIL 開啟並轉 RGB），轉為 NumPy 陣列 (HxWx3 uint8)，
        (2)回傳列表。
    •convert_label_to_1ch(label_rgb_array)：
        (1)將單張彩色標籤圖（HxWx3 uint8）轉為單通道遮罩（HxW uint8），
        (2)其中彩色映射至 0–5 六種類別索引，其餘為 255。
        (3)具體做法是將標籤展平後與定義好的六種 RGB 顏色序列比對，將匹配像素設為相應類別索引。
    '''
    '''
    (1)讀取完所有影像與標籤後，得到 data_images（列表 of HxWx3 陣列）和 label_images_6ch（列表 of (6, H, W) one-hot 陣列，或選擇 (H, W, 6) 的排列；程式備註可轉置）。
    (2)接著使用 sklearn.model_selection.train_test_split，按照 percentage_training 和 percentage_test 的比例先將資料切為「訓練集 + 臨時集」與「測試集」，再將臨時集依剩餘比例切出「驗證集」。
    (3)得到的 X_train,X_temp,y_train,y_temp、X_val,X_test,y_val,y_test（此部分程式碼將 X_temp、y_temp 再分割為 X_val,y_val 和 X_test,y_test）。
    (4)接著定義內部函式 batch_data(X, y, batch_size)：將資料列表按 batch_size 切片，回傳列表，每元素為 (X_batch, y_batch)。
    (5)利用此函式將訓練、驗證、測試資料分批，得到批次列表。
    (6)最後使用 tf.data.Dataset.from_generator，將批次資料封裝成 TensorFlow 資料集(tf.data.Dataset)，同時在此封裝內部定義另一函式 pad_input(img) （若需要，對輸入影像或標籤做一圈邊界填充以符合模型需求），以及 convert_to_tf_dataset 生成實際輸入張量。
    (7)結果回傳三個 Dataset，方便訓練過程使用。
    '''
    # Définir les couleurs et construire un tableau shape (6, 1, 1, 3) pour broadcast
    label_colors = np.array([
        (64, 67, 135),
        (34, 167, 132),
        (68, 1, 84),
        (41, 120, 142),
        (253, 231, 36),
        (121, 209, 81),
    ], dtype=np.uint8)  # (6, 3)

    # Liste des fichiers
    subset = "Crosslines" if Crosslines else "Inlines"
    data_dir = os.path.join(data_folder_path, subset)
    label_dir = os.path.join(label_folder_path, subset)

    data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.png')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')])

    assert len(data_files) == len(label_files), f"Mismatch between data ({len(data_files)}) and label ({len(label_files)}) files"

    # Chargement des images
    def load_images_RGB(file_list):
        # Retourne une liste d'array uint8 HxWx3
        imgs = []
        for f in file_list:
            im = Image.open(f).convert("RGB")
            arr = np.array(im, dtype=np.uint8)
            imgs.append(arr[:, :, :3])
        return imgs

    def convert_label_to_1ch(label_rgb_array):
        """
        label_rgb_array: HxWx3 uint8
        Retourne: HxWx1 uint8 avec une valeur entière (0-5) selon la couleur détectée, 255 sinon.
        """
        h, w, _ = label_rgb_array.shape
        label_flat = label_rgb_array.reshape(-1, 3)
        out = np.full((label_flat.shape[0],), 255, dtype=np.uint8)  # 255 = valeur par défaut (aucune couleur reconnue)
        for idx, color in enumerate(label_colors):
            matches = np.all(label_flat == color, axis=1)
            out[matches] = idx
        #print(out.shape)
        return out.reshape(h, w)


    data_images = load_images_RGB(data_files)  # list of HxWx3
    label_images_rgb = load_images_RGB(label_files)  # list of HxWx3

    # Convertir tous les labels
    label_images_6ch = [convert_label_to_1ch(lbl) for lbl in label_images_rgb]  # list of HxWx6 (transposé possible)

    # Remarque : actuellement chaque label est (6, H, W) si tu préfères (H, W, 6) tu peux transposer :
    # for i in range(len(label_images_6ch)):
    #     label_images_6ch[i] = np.transpose(label_images_6ch[i], (1, 2, 0))  # H,W,6

    # Split train / temp puis test / val
    train_size = percentage_training
    test_size = percentage_test
    val_size = 1 - train_size - test_size
    assert val_size >= 0, "Les pourcentages doivent sommer à au plus 1."

    X_train, X_temp, y_train, y_temp = train_test_split(
        data_images,
        label_images_6ch,
        train_size=train_size,
        random_state=42
    )
    # Pour le split test/val : on veut que test corresponde à percentage_test
    if test_size + val_size == 0:
        raise ValueError("Somme de test et validation est 0.")
    test_ratio_within_temp = test_size / (test_size + val_size)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio_within_temp,
        random_state=42
    )

    # Batching
    def batch_data(X, y, batch_size):
        return [(X[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(X), batch_size)]

    data_train = batch_data(X_train, y_train, batch_size)
    data_val = batch_data(X_val, y_val, batch_size)
    data_test = batch_data(X_test, y_test, batch_size)

    return data_train, data_val, data_test



def easy_load_data(label,Crosslines=True):
    '''Load data from a specific label folder and split it into training, validation, and test sets.'''
    '''
    •參數：
        (1)label 為子資料夾名稱（例如 'Line001'），
        (2)Crosslines 控制使用哪個子資料夾。
    •回傳：
        與 split_data 相同，直接回傳 (train_dataset, test_dataset, val_dataset)。
    •邏輯：
        (1)內部會根據 label 與 Crosslines 參數，組合對應的 data_folder_path 與 label_folder_path，
        (2)然後呼叫上述 split_data。此函式提供使用者快速載入指定資料夾數據並拆分的便利方法。
    '''
    t,v,test = split_data(
        data_folder_path=f'/home/user/Desktop/TF_SEG2020/SEG2020/Images/{label}', 
        label_folder_path="/home/user/Desktop/TF_SEG2020/SEG2020/Images/Labels",
        batch_size=2,                        # attention size in memory
        Crosslines=Crosslines,
        percentage_training=0.7,
        percentage_test=0.2
    )

    return t,v,test




def load_tf_dataset(t, test, val):
    # （註：定義在 data_loader 檔中）
    '''returns a tf.data.Dataset for training, testing and validation data.'''
    '''
    •參數：
        預期傳入已經批次化的訓練、測試、驗證資料（例如 train_data, test_data, val_data）。
    •回傳：
        回傳三個 tf.data.Dataset，對應訓練、測試、驗證集。
    •邏輯：
        (1)定義了兩個內部函式：pad_input(img) 用於對影像或標籤進行邊界填充，確保模型輸入大小滿足要求；
        (2)convert_to_tf_dataset(batched_data) 將批次化的資料清單（每個元素是 (X_batch, y_batch)）依序產生 NumPy 陣列，回傳生成器。
        (3)最後使用 tf.data.Dataset.from_generator 將訓練、測試、驗證的批次化資料轉換為 Dataset 物件，並指定每個元素的輸出型態與形狀。  
    '''
    def pad_input(img):
        if img.ndim == 3:  # Image RGB
            return np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        elif img.ndim == 2:  # Label image
            return np.pad(img, ((1, 1), (1, 1)), mode='reflect')
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")


    def convert_to_tf_dataset(batched_data):
        for X_list, y_list in batched_data:
            X = np.stack([pad_input(x) for x in X_list]) / 255.0
            y = np.stack([pad_input(y) for y in y_list]) #/ 255.0
            yield X, y


    train_dataset = tf.data.Dataset.from_generator(
        lambda: convert_to_tf_dataset(t),
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, 1008, 592, 3), (None, 1008, 592))
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: convert_to_tf_dataset(test),
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, 1008, 592, 3), (None, 1008, 592))
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: convert_to_tf_dataset(val),
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, 1008, 592, 3), (None, 1008, 592))
    )

    
    return train_dataset, test_dataset, val_dataset
    '''
    其他內部函式
    •	load_images_RGB（見 split_data 定義）：已在上面說明，用於載入影像檔。
    •	convert_label_to_1ch（見 split_data）：將彩色標籤圖轉為單通道。
    •	pad_input, convert_to_tf_dataset（見 load_tf_dataset 定義）：用於資料集建立時處理圖像尺寸與生成張量。
    整體而言，data_loader.py 提供了一條龍數據載入流程：從檔案系統讀取影像及標籤、進行必要的格式轉換和分割，最後產出方便送入模型訓練的 tf.data.Dataset。
    '''
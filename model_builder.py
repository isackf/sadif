from tensorflow.keras import layers, models

'''
•model_builder.py：定義並建立 U-Net 架構的模型。提供 build_unet 函式，可設定初始濾波器數量與輸入尺寸，回傳未編譯的 Keras 模型。

檔案: model_builder.py	
主要用途: 建構 U-Net 模型架構	
主要函式: build_unet (建立具有跳躍連接的 U-Net，6 類輸出)

'''
'''
model_builder.py 檔案分析
此模組用於構建 U-Net 類神經網路模型。
'''

def build_unet(base=16,input_shape=(1008, 592, 3)):
    '''
    •參數：
        (1)base 為 U-Net 底層濾波器數量（整數，例如 16），
        (2)input_shape 為輸入影像形狀（高、寬、通道數）。
    •回傳：
        回傳一個未編譯的 tf.keras.Model 模型。
    •邏輯：
        (1)按照 U-Net 標準架構，首先建立輸入層。
        (2)然後在編碼器（downsampling）部分依次進行卷積和池化：
        (3)第一層兩次 (3×3) 卷積（base 個濾波器，ReLU 激活），再一次池化；
        (4)第二層用 base*2 個濾波器，再池化；第三層用 base*4 個，再池化，以此類推。
        (5)本程式碼顯示了最多到第三層的上採樣（deconvolution）部分：在解碼器段先進行上採樣，與對應編碼器層做 Concatenate，再經過卷積。
        (6)最後輸出層使用 1×1 卷積產生 6 個通道（對應 6 類別）並以 softmax 激活。
        (7)由於是分割模型，每個位置輸出對應 6 類的機率。模型建立後回傳，供訓練或預測使用（註：此函式未在內部呼叫 compile）。
    '''
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(base, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(base, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(base*2, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(base*2, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(base*4, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(base*4, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    b = layers.Conv2D(base*8, (3, 3), activation='relu', padding='same')(p3)
    b = layers.Conv2D(base*8, (3, 3), activation='relu', padding='same')(b)

    # Decoder
    u3 = layers.UpSampling2D((2, 2))(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(base*4, (3, 3), activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(base*4, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(base*2, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(base*2, (3, 3), activation='relu', padding='same')(c5)

    u1 = layers.UpSampling2D((2, 2))(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(base, (3, 3), activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(base, (3, 3), activation='relu', padding='same')(c6)

    # c7 = layers.Conv2D(3, (1, 1), activation='relu', padding = 'same')(c6)
    outputs = layers.Conv2D(6, (1, 1), activation='softmax')(c6)

    model = models.Model(inputs, outputs)
    return model

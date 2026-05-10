

def reconhece_animais(caminho:str) -> str:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import numpy as np
    import pandas as pd
    import os

    model_k = tf.keras.models.load_model('modelh5.keras')

    img = image.load_img(caminho, target_size=(180, 180))
    img_array = image.img_to_array(img)

    # normalização

    img_array = np.expand_dims(img_array, axis=0)

    pred = model_k.predict(img_array)

    class_names = os.listdir("C:\\Users\\Nitro\\PyCharmMiscProject\\Projetos_DeepLearning\\animais_dataset\\treino")
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    return f"Classe: {pred_class}, Confidence: {confidence:.2%}"

if __name__ == "__main__":
    animal = reconhece_animais("C:\\Users\\Nitro\\Downloads\\Captura de tela 2026-05-09 231337.png")
    print(animal)


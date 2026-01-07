import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Konfigurasi Global 
IMG_HEIGHT, IMG_WIDTH = 224, 224
MODEL_PATH = "best_model_resnet50.keras" 
CLASS_NAMES_PATH = 'class_names.npy'

# Variabel global untuk menyimpan model dan nama kelas 
_model = None
_class_names = None

def load_model_for_prediction():
    """Memuat model dan nama kelas jika belum dimuat."""
    global _model, _class_names
    if _model is None:
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            _class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
            print(f"Model {MODEL_PATH} dan nama kelas dimuat berhasil untuk prediksi.")
        except Exception as e:
            print(f"Error memuat model atau nama kelas: {e}")
            _model = None
            _class_names = None
    return _model, _class_names

def predict_food(image: Image.Image):
    """
    Melakukan prediksi kelas makanan pada gambar yang diberikan.

    Args:
        image (PIL.Image.Image): Objek gambar PIL yang akan diprediksi.

    Returns:
        tuple: (predicted_class_name, confidence)
    """
    model, class_names = load_model_for_prediction()
    if model is None or class_names is None:
        raise RuntimeError("Model atau nama kelas gagal dimuat.")

    # Pra-pemrosesan gambar: resize dan konversi ke array numpy
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Menambah dimensi batch

    # Pra-pemrosesan khusus untuk ResNet50 
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, confidence

if __name__ == '__main__':
    from PIL import Image
    try:
        test_image_path = 'path/to/your/test_image.jpg' 
        img = Image.open(test_image_path)
        predicted_class, confidence = predict_food(img)
        print(f"Prediksi: {predicted_class}, Kepercayaan: {confidence:.2f}%")
    except FileNotFoundError:
        print(f"Error: File gambar tidak ditemukan di {test_image_path}")
    except Exception as e:
        print(f"Terjadi kesalahan saat prediksi: {e}")
    print("Script ini hanya untuk fungsi prediksi. Jalankan train_and_evaluate_model.py untuk pelatihan.")
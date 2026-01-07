import os
# Mengatur variabel lingkungan untuk TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Menekan peringatan TensorFlow

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50 # Mengganti MobileNetV2 dengan ResNet50
import numpy as np
import matplotlib.pyplot as plt # Dipertahankan untuk visualisasi training history
import seaborn as sns # Dipertahankan untuk visualisasi confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
# Menghapus tqdm, TqdmCallback, dan colorama karena menggunakan progress bar standar Keras

# Konfigurasi Global
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 64
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
DATA_DIR = "dataset"
# Mengubah nama file model untuk mencerminkan penggunaan ResNet50
MODEL_PATH = "food_classifier_model_resnet50.h5" 
CLASS_NAMES_PATH = 'class_names.npy'

# Fungsi untuk memuat dan memproses dataset
def prepare_data():
    AUTOTUNE = tf.data.AUTOTUNE
    # Memuat dataset pelatihan dan validasi dari direktori
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, 
        validation_split=0.2, 
        subset="training", 
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        batch_size=BATCH_SIZE
    ).cache().shuffle(1000).prefetch(AUTOTUNE)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, 
        validation_split=0.2, 
        subset="validation", 
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        batch_size=BATCH_SIZE
    ).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds

# Fungsi untuk analisis distribusi kelas dalam dataset
def analyze_class_distribution(data_dir):
    # Mendapatkan nama-nama kelas dari subdirektori
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if len(class_names) < 2:
        raise ValueError("Dataset harus memiliki setidaknya dua kelas.")
    
    # Menghitung jumlah gambar per kelas
    class_counts = {name: len(os.listdir(os.path.join(data_dir, name))) for name in class_names}
    print("Distribusi kelas:", class_counts)
    
    return class_names, class_counts

# Fungsi untuk menghitung bobot kelas untuk mengatasi ketidakseimbangan data
def compute_class_weights(class_counts):
    # Membuat daftar label berdasarkan jumlah sampel per kelas
    labels = [idx for idx, count in enumerate(class_counts.values()) for _ in range(count)]
    # Menghitung bobot kelas menggunakan sklearn.utils.class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(class_weights))

# Fungsi untuk membuat dan mengkompilasi model pembelajaran mendalam
def build_model(num_classes):
    # Lapisan augmentasi data untuk meningkatkan variasi dataset
    data_augmentation = models.Sequential([
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal_and_vertical"), 
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2)
    ], name="data_augmentation_layer") # Memberi nama lapisan untuk akses mudah

    # Menggunakan ResNet50 sebagai model dasar (pre-trained di ImageNet)
    # include_top=False: tidak menyertakan lapisan klasifikasi atas
    base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Membekukan lapisan model dasar pada awalnya

    # Membangun arsitektur model
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs) # Menerapkan augmentasi data
    # Preprocessing input khusus untuk ResNet50, pastikan data sesuai
    x = tf.keras.applications.resnet50.preprocess_input(x) # Tambahkan ini di sini
    x = base_model(x, training=False) # Meneruskan melalui model dasar (dalam mode inference)
    x = layers.GlobalAveragePooling2D()(x) # Mengurangi dimensi spasial
    
    # Lapisan dense kustom untuk klasifikasi
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x) # Lapisan output dengan aktivasi softmax

    model = models.Model(inputs, outputs, name="food_classifier_resnet50")

    # Mengkompilasi model untuk pelatihan awal
    # Menggunakan Adam optimizer dengan learning rate default (0.001)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    model.build((None, IMG_HEIGHT, IMG_WIDTH, 3)) # Membangun model dengan shape input yang ditentukan
    return model

# Fungsi untuk mengevaluasi kinerja model
def evaluate_model(model, val_ds, class_names):
    y_true_list = []
    y_pred_list = []
    
    # Kumpulkan semua label sebenarnya dari dataset validasi
    for _, labels in val_ds.unbatch().as_numpy_iterator():
        y_true_list.append(labels)

    # Buat single tensor dari seluruh dataset validasi untuk prediksi efisien
    val_images_list = []
    for images, _ in val_ds.unbatch().as_numpy_iterator():
        val_images_list.append(images)
    
    # Penting: Lakukan preprocessing input di sini juga untuk evaluasi
    val_images_processed = tf.keras.applications.resnet50.preprocess_input(np.array(val_images_list))
    
    # Prediksi semua gambar validasi sekaligus
    predictions = model.predict(val_images_processed)
    y_pred_list = np.argmax(predictions, axis=1)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Menghitung confusion matrix
    return y_true, y_pred, confusion_matrix(y_true, y_pred)

# Fungsi utama untuk melatih model
def train_model_resnet():
    # Persiapan data
    train_ds, val_ds = prepare_data()
    class_names, class_counts = analyze_class_distribution(DATA_DIR)
    np.save(CLASS_NAMES_PATH, class_names) # Menyimpan nama-nama kelas
    class_weight = compute_class_weights(class_counts)
    print("Class weights:", class_weight)

    # Membangun dan mengkompilasi model
    model = build_model(num_classes=len(class_names))
    model.summary()

    # Callback untuk pelatihan
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True), # Kesabaran ditingkatkan
        tf.keras.callbacks.ModelCheckpoint('best_model_resnet50.h5', monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1) # Kesabaran ditingkatkan
    ]
    
    print("\n--- Memulai Pelatihan Awal (Feature Extraction) ---")
    # Melatih model (verbose=1 untuk menampilkan progress standar)
    history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, 
                        class_weight=class_weight, callbacks=callbacks, verbose=1) # verbose=1 untuk progress standar

    # Fase Fine-tuning
    print("\n--- Memulai Fine-tuning ---")
    # Mengakses model dasar ResNet50 dari lapisan model utama
    # base_model adalah lapisan kedua setelah input dan augmentasi
    base_model = model.get_layer("resnet50") # Akses berdasarkan nama lapisan ResNet50
    base_model.trainable = True # Membuka blokir lapisan model dasar

    # Membekukan lapisan-lapisan awal dari ResNet50, hanya melatih lapisan-lapisan akhir
    # ResNet50 memiliki sekitar 175 lapisan. Membuka blokir 50 lapisan terakhir adalah titik awal yang baik.
    for layer in base_model.layers[:-10]: 
        layer.trainable = False

    # Mengkompilasi ulang model untuk menerapkan perubahan trainable
    # Menggunakan learning rate yang jauh lebih rendah untuk fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Learning rate lebih rendah untuk fine-tuning
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Melanjutkan pelatihan dengan fine-tuning
    history_fine = model.fit(train_ds, validation_data=val_ds, 
                             epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
                             initial_epoch=history.epoch[-1] + 1, # Melanjutkan dari epoch terakhir pelatihan awal
                             class_weight=class_weight, callbacks=callbacks, verbose=1) # verbose=1 untuk progress standar

    # Evaluasi akhir model dan penyimpanan
    print("\n--- Evaluasi Model ---")
    y_true, y_pred, cm = evaluate_model(model, val_ds, class_names)
    model.save(MODEL_PATH)
    print(f"Model disimpan di {MODEL_PATH}")

    return history, history_fine, class_names, y_true, y_pred, cm

# Blok ini akan dieksekusi hanya jika script ini dijalankan secara langsung (mis. untuk pelatihan)
if __name__ == "__main__":
    # Melatih model saat script dijalankan langsung
    history, history_fine, class_names, y_true, y_pred, cm = train_model()

    # Bagian visualisasi (menggunakan matplotlib dan seaborn)
    print("\n--- Membuat Visualisasi Hasil ---")
    # Plot Distribusi Kelas
    class_counts = {name: len(os.listdir(os.path.join(DATA_DIR, name))) for name in class_names}
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Distribusi Kelas dalam Dataset")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=45, ha='right') # Memutar label X untuk keterbacaan
    plt.tight_layout() # Menyesuaikan layout
    plt.savefig("class_distribution.png")
    plt.close()

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8)) # Ukuran sedikit lebih besar
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right') # Memutar label X
    plt.yticks(rotation=0) # Memastikan label Y tidak terputar
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Plot Riwayat Pelatihan (Akurasi dan Loss)
    plt.figure(figsize=(16, 6)) # Ukuran figure lebih besar

    # Subplot Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy')

    # Menandai awal fase fine-tuning
    fine_tune_start_epoch = len(history.history['accuracy'])
    plt.axvline(x=fine_tune_start_epoch, color='r', linestyle='--', label='Fine-tune Start') 
    # Menempatkan teks "Fine-tuning"
    max_acc = max(max(history.history['accuracy'] + history_fine.history['accuracy']), 
                  max(history.history['val_accuracy'] + history_fine.history['val_accuracy']))
    plt.text(fine_tune_start_epoch + 0.5, max_acc * 0.9, 'Fine-tuning', color='r', rotation=90, va='top', ha='left') 

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True) # Menambahkan grid

    # Subplot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss')

    # Menandai awal fase fine-tuning
    plt.axvline(x=fine_tune_start_epoch, color='r', linestyle='--', label='Fine-tune Start')
    # Menempatkan teks "Fine-tuning"
    max_loss = max(max(history.history['loss'] + history_fine.history['loss']), 
                   min(history.history['val_loss'] + history_fine.history['val_loss'])) # Gunakan min untuk lower bound teks
    plt.text(fine_tune_start_epoch + 0.5, max_loss * 0.9, 'Fine-tuning', color='r', rotation=90, va='top', ha='left')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True) # Menambahkan grid

    plt.tight_layout() # Menyesuaikan layout
    plt.savefig('training_history.png')
    plt.close()
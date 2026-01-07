import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import shutil
from datetime import datetime


IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 18
FINE_TUNE_EPOCHS = 12
DATA_DIR = "dataset_cleaned1"
MODEL_PATH = "food_classifier_model_resnet50.keras"  
CLASS_NAMES_PATH = 'class_names.npy'

def load_data_paths_and_labels(data_dir):
    all_image_paths = []
    all_image_labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    if len(class_names) < 2:
        raise ValueError("Dataset harus memiliki setidaknya dua kelas.")

    label_to_index = {name: i for i, name in enumerate(class_names)}

    print("Memuat jalur gambar dan label...")
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        images_in_class = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        all_image_paths.extend(images_in_class)
        all_image_labels.extend([label_to_index[class_name]] * len(images_in_class))
    
    all_image_paths = np.array(all_image_paths)
    all_image_labels = np.array(all_image_labels)

    return all_image_paths, all_image_labels, class_names

def preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label
   

def prepare_data_stratified():
    AUTOTUNE = tf.data.AUTOTUNE

    all_image_paths, all_image_labels, class_names = load_data_paths_and_labels(DATA_DIR)
    
    X_train, X_val, y_train, y_val = train_test_split(
        all_image_paths, all_image_labels,
        test_size=0.2,
        stratify=all_image_labels,
        random_state=123
    )

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    print(f"Jumlah sampel training: {len(X_train)}")
    print(f"Jumlah sampel validation: {len(X_val)}")

    return train_ds, val_ds, class_names, all_image_labels

# --- Fungsi analyze_class_distribution disesuaikan ---
def analyze_class_distribution(data_dir):
    class_names_local = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_counts = {name: len(os.listdir(os.path.join(data_dir, name))) for name in class_names_local}
    
    print("Distribusi kelas:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    
    return class_names_local, class_counts

# --- Fungsi untuk menghitung bobot kelas ---
def compute_class_weights(class_names, all_image_labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(all_image_labels), y=all_image_labels)
    return dict(enumerate(class_weights))  # Ensure it's a simple dict of floats

def build_model(num_classes):
    data_augmentation = models.Sequential([
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal_and_vertical"), 
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2)
    ], name="data_augmentation_layer")

    base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="food_classifier_resnet50")

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(model, val_ds, class_names):
    y_true_list = []
    y_pred_list = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, batch_size=BATCH_SIZE)
        y_pred_list.extend(np.argmax(predictions, axis=1))
        y_true_list.extend(labels.numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    print("\n------------- Classification Report ---------------")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n--- Per-Class Accuracy ---")
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        true_indices = np.where(y_true == i)
        if len(true_indices[0]) > 0:
            correct_predictions = np.sum(y_pred[true_indices] == i)
            total_samples = len(true_indices[0])
            per_class_accuracy[class_name] = correct_predictions / total_samples
            print(f"{class_name}: {per_class_accuracy[class_name]:.4f}")
        else:
            per_class_accuracy[class_name] = 0.0
            print(f"{class_name}: Tidak ada sampel di validasi")

    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, cm

# --- Fungsi utama untuk melatih model ---
def train_model():
    train_ds, val_ds, class_names, all_image_labels = prepare_data_stratified()
    np.save(CLASS_NAMES_PATH, class_names)
    
    class_weight = compute_class_weights(class_names, all_image_labels)
    print("Class weights:", class_weight)

    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model_resnet50.keras', monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    print("\n--- Memulai Pelatihan Awal (Feature Extraction) ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, class_weight=class_weight, callbacks=callbacks, verbose=1)

    print("\n--- Memulai Fine-tuning ---")
    base_model = model.get_layer("resnet50")
    base_model.trainable = True

    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history_fine = model.fit(train_ds, validation_data=val_ds,
                             epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
                             initial_epoch=history.epoch[-1] + 1,
                             class_weight=class_weight, callbacks=callbacks, verbose=1)

    print("\n--- Evaluasi Model ---")
    y_true, y_pred, cm = evaluate_model(model, val_ds, class_names)
    model.save(MODEL_PATH)
    print(f"Model disimpan di {MODEL_PATH}")

    return history, history_fine, class_names, y_true, y_pred, cm, train_ds, val_ds

if __name__ == "__main__":
    history, history_fine, class_names, y_true, y_pred, cm, train_ds, val_ds = train_model()

    print("\n--- Membuat Visualisasi Hasil ---")
    class_counts_for_plot = {name: len(os.listdir(os.path.join(DATA_DIR, name))) for name in class_names}

    plt.figure(figsize=(12, 6))
    plt.bar(class_counts_for_plot.keys(), class_counts_for_plot.values())
    plt.title("Distribusi Kelas dalam Dataset")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=90, ha='center')
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy')
    fine_tune_start_epoch = len(history.history['accuracy'])
    plt.axvline(x=fine_tune_start_epoch, color='r', linestyle='--', label='Fine-tune Start')
    max_acc = max(max(history.history['accuracy'] + history_fine.history['accuracy']),
                  max(history.history['val_accuracy'] + history_fine.history['val_accuracy']))
    plt.text(fine_tune_start_epoch + 0.5, max_acc * 0.9, 'Fine-tuning', color='r', rotation=90, va='top', ha='left')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss')
    plt.axvline(x=fine_tune_start_epoch, color='r', linestyle='--', label='Fine-tune Start')
    max_loss = max(max(history.history['loss'] + history_fine.history['loss']),
                   min(history.history['val_loss'] + history_fine.history['val_loss']))
    plt.text(fine_tune_start_epoch + 0.5, max_loss * 0.9, 'Fine-tuning', color='r', rotation=90, va='top', ha='left')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    # ===================================================================
    # TAMBAHAN: Augmentasi Otomatis untuk Kelas Precision < 0.75
    # ===================================================================

    from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

    print("\n" + "="*80)
    print("PERBAIKAN OTOMATIS: KELAS DENGAN PRECISION < 0.75")
    print("="*80)

    # --- 1. Backup dataset sebelum augmentasi ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"dataset_cleaned1_backup_{timestamp}"
    if not os.path.exists(backup_dir):
        print(f"Backup dataset asli ke: {backup_dir}")
        shutil.copytree(DATA_DIR, backup_dir)
    else:
        print(f"Backup sudah ada: {backup_dir}")

    # --- 2. Hitung precision per kelas ---
    precision_per_class = {}
    for i, class_name in enumerate(class_names):
        true_positives = np.sum((y_true == i) & (y_pred == i))
        predicted_positives = np.sum(y_pred == i)
        if predicted_positives > 0:
            precision_per_class[class_name] = true_positives / predicted_positives
        else:
            precision_per_class[class_name] = 0.0

    # --- 3. Deteksi kelas lemah ---
    low_precision_classes = {k: v for k, v in precision_per_class.items() if v < 0.80}
    
    if low_precision_classes:
        print(f"Ditemukan {len(low_precision_classes)} kelas dengan precision < 0.75:")
        for cls, prec in low_precision_classes.items():
            count = len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  → {cls}: precision = {prec:.3f}, support = {count} gambar")
    else:
        print("Semua kelas memiliki precision ≥ 0.75. Tidak perlu perbaikan.")
        low_precision_classes = {}

    # --- 4. Augmentasi otomatis LANGSUNG ke folder asli ---
    if low_precision_classes:
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )

        target_count = 60  # Target minimal gambar per kelas

        for class_name, _ in low_precision_classes.items():
            class_dir = os.path.join(DATA_DIR, class_name)
            current_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            needed = max(0, target_count - current_count)

            if needed > 0:
                print(f"  → Augmentasi {class_name}: {current_count} → {current_count + needed} gambar")
                img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                augment_count = 0
                while augment_count < needed:
                    img_path = os.path.join(class_dir, random.choice(img_files))
                    img = load_img(img_path)
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)

                    i = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=class_dir,
                                            save_prefix=f"aug_{class_name}", save_format='jpeg'):
                        i += 1
                        augment_count += 1
                        if i >= 3 or augment_count >= needed:
                            break
                    if augment_count >= needed:
                        break
            else:
                print(f"  → {class_name}: Sudah cukup ({current_count} gambar)")

        print(f"\nGambar augmentasi berhasil disimpan LANGSUNG ke folder asli: {DATA_DIR}")
        print("Anda bisa langsung jalankan training ulang tanpa ubah apapun!")

    else:
        print("Tidak ada augmentasi diperlukan.")

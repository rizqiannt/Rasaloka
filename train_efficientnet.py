import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import random

# --- Konfigurasi ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
DATA_DIR = "dataset_cleaned1"
MODEL_PATH = "food_classifier_model_efficientnet.keras"
BEST_MODEL_PATH = "best_model_efficientnet.keras"
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
        images_in_class = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir) 
                           if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        all_image_paths.extend(images_in_class)
        all_image_labels.extend([label_to_index[class_name]] * len(images_in_class))
    
    all_image_paths = np.array(all_image_paths)
    all_image_labels = np.array(all_image_labels)

    return all_image_paths, all_image_labels, class_names

def load_and_preprocess_image(path, label, num_classes):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # EfficientNet preprocess_input (biasanya scaling 0-255 atau -1 ke 1 tergantung model,
    # tapi tf.keras.applications.efficientnet menangani ini dgn benar)
    img = preprocess_input(img) 
    return img, tf.one_hot(label, depth=num_classes)

def prepare_data_stratified():
    AUTOTUNE = tf.data.AUTOTUNE

    all_image_paths, all_image_labels, class_names = load_data_paths_and_labels(DATA_DIR)
    num_classes = len(class_names)
    
    X_train, X_val, y_train, y_val = train_test_split(
        all_image_paths, all_image_labels,
        test_size=0.2,
        stratify=all_image_labels,
        random_state=123
    )

    # Wrapper agar bisa pass num_classes ke map function
    def preprocess_wrapper(path, label):
        return load_and_preprocess_image(path, label, num_classes)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(preprocess_wrapper, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(preprocess_wrapper, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    print(f"Jumlah sampel training: {len(X_train)}")
    print(f"Jumlah sampel validation: {len(X_val)}")

    return train_ds, val_ds, class_names, all_image_labels

def compute_class_weights(class_names, all_image_labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(all_image_labels), y=all_image_labels)
    return dict(enumerate(class_weights))

def build_model(num_classes):
    data_augmentation = models.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # EfficientNetB1
    base_model = EfficientNetB1(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="food_classifier_efficientnet")

    # Menggunakan Label Smoothing untuk mengurangi overfitting dan meningkatkan generalisasi
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

import json
from datetime import datetime

# ... (Previous imports and configs remain valid)

def evaluate_model_report(model, val_ds, class_names):
    y_true_list = []
    y_pred_list = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
        y_pred_list.extend(np.argmax(predictions, axis=1))
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    
    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, cm, report_dict, report_str

def train_model(fine_tune_epochs=FINE_TUNE_EPOCHS, learning_rate=1e-5):
    train_ds, val_ds, class_names, all_image_labels = prepare_data_stratified()
    np.save(CLASS_NAMES_PATH, class_names)
    
    class_weight = compute_class_weights(class_names, all_image_labels)
    print("Class weights:", class_weight)

    model = build_model(num_classes=len(class_names))
    # model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    print("\n--- Memulai Pelatihan Awal (Feature Extraction) ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, 
                        class_weight=class_weight, callbacks=callbacks, verbose=1)

    print(f"\n--- Memulai Fine-tuning (Epochs: {fine_tune_epochs}, LR: {learning_rate}) ---")
    
    # Mencari layer EfficientNet di dalam model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            for sub_layer in layer.layers[:-50]:
                sub_layer.trainable = False
            print(f"Unfreezing deep layers of {layer.name}")

    # Recompile dengan learning rate dari parameter
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    total_epochs =  len(history.history['loss']) + fine_tune_epochs

    history_fine = model.fit(train_ds, validation_data=val_ds,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             class_weight=class_weight, callbacks=callbacks, verbose=1)

    print("\n--- Evaluasi Model ---")
    y_true, y_pred, cm, report_dict, report_str = evaluate_model_report(model, val_ds, class_names)
    
    model.save(MODEL_PATH)
    print(f"Model disimpan di {MODEL_PATH}")

    # Simpan log ke JSON
    final_accuracy = history_fine.history['accuracy'][-1]
    final_loss = history_fine.history['loss'][-1]
    
    log_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": fine_tune_epochs,
        "learning_rate": learning_rate,
        "accuracy": f"{final_accuracy:.4f}",
        "loss": f"{final_loss:.4f}",
        "classification_report": report_dict
    }

    log_file = "training_log.json"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    else:
        logs = []
        
    logs.insert(0, log_entry) # Insert at beginning
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

    return history, history_fine, class_names, y_true, y_pred, cm, report_str

if __name__ == "__main__":
    history, history_fine, class_names, y_true, y_pred, cm, report_str = train_model()

    print("\n--- Membuat Visualisasi Hasil ---")
    
    # Gabungkan history
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history_efficientnet.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (EfficientNet)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_efficientnet.png")
    plt.close()

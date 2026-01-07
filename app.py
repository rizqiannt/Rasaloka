import os
from flask import Flask, render_template, request, url_for, redirect, jsonify
from predict_model import predict_food
from PIL import Image
import shutil
import logging
import time
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'supersecretkey' # Required for session management

from admin import admin_bp
app.register_blueprint(admin_bp)

# Pastikan folder uploads dan static/uploads ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['STATIC_UPLOAD_FOLDER']):
    os.makedirs(app.config['STATIC_UPLOAD_FOLDER'])

# Dictionary untuk memetakan makanan ke asal daerah
REGION_MAPPING = {
    'rendang': 'PADANG',
    'pempek': 'PALEMBANG',
    'gudeg': 'DI YOGYAKARTA',
    'ayam betutu': 'BALI',
    'ayam taliwang': 'NTB',
    'bika ambon': 'MEDAN',
    'coto': 'MAKASSAR',
    'empal gentong': 'CIREBON',
    'papeda': 'PAPUA & MALUKU',
    'rawon': 'SURABAYA',
    # 'sate maranggi': 'PURWAKARTA',
    'telor asin': 'BREBES',
    'kerak telor': 'DKI JAKARTA',
    'gulai belacan' : 'RIAU'
}

MIN_CONFIDENCE = 85.0  # Threshold minimal confidence untuk menganggap sebagai makanan yang valid

def remove_file_safely(filepath):
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.debug(f"File {filepath} berhasil dihapus.")
                break
        except PermissionError:
            logging.warning(f"Attempt {attempt + 1} failed: File {filepath} is in use. Retrying...")
            time.sleep(1)
    else:
        logging.error(f"Gagal menghapus {filepath} setelah {max_attempts} percobaan.")

@app.route('/')
def index():
    logging.debug("Accessing landing page")
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    logging.debug("Accessing predict route")
    result = None
    image_url = None
    image_filename = None
    error = None

    if request.method == 'POST':
        logging.debug("POST request received")

        # Handle realtime prediction via JSON (base64 image)
        if request.is_json:
            data = request.get_json()
            if 'image' in data:
                try:
                    img_data = base64.b64decode(data['image'].split(',')[1])
                    img = Image.open(BytesIO(img_data))
                    predicted_class, confidence = predict_food(img)
                    predicted_class_lower = predicted_class.lower()
                    if predicted_class_lower not in REGION_MAPPING or confidence < MIN_CONFIDENCE:
                        return jsonify({'error': 'Gambar bukan makanan yang dikenali. Silakan masukkan gambar makanan.'}), 400
                    region = REGION_MAPPING.get(predicted_class_lower, "Tidak diketahui")
                    result = {
                        'name': predicted_class.upper(),
                        'region': region,
                        'confidence': f"{confidence:.2f}%"
                    }
                    return jsonify(result)
                except Exception as e:
                    logging.error(f"Error in realtime prediction: {str(e)}")
                    return jsonify({'error': str(e)}), 500

        # Handle file upload (fallback from gallery)
        action = request.form.get('action', 'upload')
        logging.debug(f"Action received: {action}")

        if action == 'delete':
            filename = request.form.get('filename')
            if filename:
                filepath = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], filename)
                remove_file_safely(filepath)
            return redirect(url_for('predict'))

        elif action == 'upload':
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    error = "No selected file"
                elif file:
                    # Hapus file lama
                    for old_file in os.listdir(app.config['STATIC_UPLOAD_FOLDER']):
                        old_filepath = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], old_file)
                        remove_file_safely(old_filepath)
                    
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filepath)
                    static_filepath = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], file.filename)
                    shutil.copy(filepath, static_filepath)
                    image_url = url_for('static', filename=f'uploads/{file.filename}')
                    image_filename = file.filename

                    # Langsung prediksi dari file yang diupload
                    try:
                        with Image.open(static_filepath) as img:
                            predicted_class, confidence = predict_food(img)
                        predicted_class_lower = predicted_class.lower()
                        if predicted_class_lower not in REGION_MAPPING or confidence < MIN_CONFIDENCE:
                            error = 'Gambar bukan makanan. Silakan masukkan gambar makanan.'
                            # Hapus file jika bukan makanan
                            remove_file_safely(static_filepath)
                            image_url = None
                            image_filename = None
                        else:
                            region = REGION_MAPPING.get(predicted_class_lower, "Tidak diketahui")
                            result = {
                                'name': predicted_class.upper(),
                                'region': region,
                                'confidence': f"{confidence:.2f}%"
                            }
                    except Exception as e:
                        error = f"Error processing image: {str(e)}"

                    # Hapus file sementara
                    remove_file_safely(filepath)

    return render_template(
        'predict.html',
        result=result,
        image_url=image_url,
        image_filename=image_filename,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
import threading
import logging
import json
import os
from train_efficientnet import train_model
from train_resnet import train_model_resnet

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Hardcoded credentials for demonstration
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin1'

# Global flag to track training status
is_training = False
training_thread = None

# Configure upload path within the blueprint context, but relative to app root
DATASET_DIR = 'dataset_cleaned1'

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('admin.login'))
            
    return render_template('admin_login.html')

@admin_bp.route('/dashboard')
def dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.login'))
    
    # Load training logs
    logs = []
    log_file = "training_log.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
            
    return render_template('admin_dashboard.html', logs=logs)

@admin_bp.route('/train', methods=['POST'])
def train():
    global is_training, training_thread
    
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.login'))
        
    if is_training:
        flash('Training is already in progress.', 'warning')
    else:
        try:
            epochs = int(request.form.get('epochs', 15))
            lr_str = request.form.get('learning_rate', '0.0001')
            learning_rate = float(lr_str)
            
            is_training = True
            training_thread = threading.Thread(target=run_training_background, args=(epochs, learning_rate))
            training_thread.start()
            flash(f'Training started in background ({epochs} epochs, LR: {learning_rate}).', 'success')
        except ValueError:
            flash('Invalid parameters for training.', 'danger')
        
    return redirect(url_for('admin.dashboard'))

def run_training_background(epochs, learning_rate):
    global is_training
    try:
        logging.info(f"Starting background training with Epochs={epochs}, LR={learning_rate}...")
        # Call the training function
        train_model(fine_tune_epochs=epochs, learning_rate=learning_rate)
        logging.info("Background training completed successfully.")
    except Exception as e:
        logging.error(f"Error during background training: {e}")
    finally:
        is_training = False

@admin_bp.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('admin.login'))


@admin_bp.route('/upload_data', methods=['POST'])
def upload_data():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.login'))

    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('admin.dashboard'))
        
    file = request.files['file']
    class_name = request.form.get('class_name')
    
    if file.filename == '' or not class_name:
        flash('No selected file or class name missing', 'danger')
        return redirect(url_for('admin.dashboard'))

    if file:
        # Sanitize class name
        class_name = "".join([c for c in class_name if c.isalnum() or c in (' ', '_')]).strip()
        target_dir = os.path.join(DATASET_DIR, class_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        filename = file.filename
        file.save(os.path.join(target_dir, filename))
        flash(f'Image uploaded successfully to class "{class_name}"', 'success')
        
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/status')
def status():
    global is_training
    return jsonify({'is_training': is_training})

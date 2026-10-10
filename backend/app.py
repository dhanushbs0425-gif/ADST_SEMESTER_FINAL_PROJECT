"""
SeedSense AI - Flask Web Application Controller
"""

import os
import json
import joblib
import torch
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, flash

from backend.config import (
    BASE_DIR, UPLOADS_DIR, RESULTS_DIR, SECRET_KEY,
    SPECIES_MODEL_PATH, QUALITY_MODEL_PATH, DEFECT_MODEL_PATH, ML_MODELS_PATH
)
from backend.analyzer import analyze_seed_image
from backend.database import save_analysis, get_history, get_analysis_by_id
from backend.report import generate_pdf_report

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32 MB max upload limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No image selected', 'warning')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            model_type = request.form.get('model_type', 'dl').lower()
            
            # Save uploaded file
            filename = f"upload_{os.urandom(4).hex()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Run analyzer
                analysis_result = analyze_seed_image(filepath, model_type=model_type)
                
                # Save to database
                record_id = save_analysis(analysis_result)
                
                return redirect(url_for('results', record_id=record_id))
            except Exception as e:
                flash(f"Error analyzing image: {str(e)}", 'danger')
                return redirect(request.url)
        else:
            flash('Unsupported file extension. Upload JPG, PNG, or WEBP.', 'danger')
            return redirect(request.url)
            
    return render_template('upload.html')

@app.route('/results/<int:record_id>')
def results(record_id):
    analysis_data = get_analysis_by_id(record_id)
    if not analysis_data:
        flash('Analysis record not found', 'danger')
        return redirect(url_for('history'))
        
    # Parse JSON fields
    if isinstance(analysis_data.get('species_distribution'), str):
        analysis_data['species_distribution'] = json.loads(analysis_data['species_distribution'])
    if isinstance(analysis_data.get('defect_distribution'), str):
        analysis_data['defect_distribution'] = json.loads(analysis_data['defect_distribution'])
        
    return render_template('results.html', data=analysis_data)

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        return upload()
    return render_template('batch.html')

@app.route('/history')
def history():
    records = get_history()
    return render_template('history.html', records=records)

@app.route('/performance')
def performance():
    # Load actual ML metrics
    ml_data = {}
    if os.path.exists(ML_MODELS_PATH):
        try:
            raw_ml = joblib.load(ML_MODELS_PATH)
            ml_data = {
                'species_results': raw_ml.get('species_results', {}),
                'quality_results': raw_ml.get('quality_results', {}),
                'best_species': raw_ml.get('best_species_classifier', 'SVM'),
                'best_quality': raw_ml.get('best_quality_classifier', 'Random Forest')
            }
        except Exception as e:
            ml_data = {'error': str(e)}
            
    # Load actual DL metrics
    dl_data = {}
    if os.path.exists(SPECIES_MODEL_PATH):
        try:
            sp_ckpt = torch.load(SPECIES_MODEL_PATH, map_location='cpu')
            q_ckpt = torch.load(QUALITY_MODEL_PATH, map_location='cpu')
            def_ckpt = torch.load(DEFECT_MODEL_PATH, map_location='cpu')
            
            dl_data = {
                'species_model': sp_ckpt.get('results', {}),
                'quality_model': q_ckpt.get('results', {}),
                'defect_model': def_ckpt.get('results', {})
            }
        except Exception as e:
            dl_data = {'error': str(e)}
            
    return render_template('performance.html', ml_data=ml_data, dl_data=dl_data)

@app.route('/dataset_info')
def dataset_info():
    report_path = os.path.join(BASE_DIR, 'dataset', 'dataset_report.json')
    report_data = {}
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_data = json.load(f)
    return render_template('dataset_info.html', report=report_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/download_pdf/<int:record_id>')
def download_pdf(record_id):
    analysis_data = get_analysis_by_id(record_id)
    if not analysis_data:
        flash('Record not found', 'danger')
        return redirect(url_for('history'))
        
    pdf_filename = f"SeedSense_Report_#{record_id}.pdf"
    pdf_path = os.path.join(RESULTS_DIR, pdf_filename)
    
    generate_pdf_report(analysis_data, pdf_path)
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

import os
import subprocess
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file
from threading import Thread, Event

app = Flask(__name__)
progress = {
    "status": "idle",
    "percent": 0,
    "last_line": "",
    "message": "",
    "start_time": None,
    "elapsed_time": 0,
}

inference_process = None
stop_event = Event()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/baseline_calculation')
def baseline_calculation():
    return render_template('baseline_calculation.html')

@app.route('/stiffness_calculation')
def stiffness_calculation():
    return render_template('stiffness_calculation.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return 'no file part'
    
    file = request.files['video']
    if file.filename == '':
        return 'no selected file'
    
    if file and allowed_file(file.filename):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        upload_folder = os.path.join(project_root, 'TransUNet', 'datasets', 'videos')
        print(f"Intentando guardar en: {upload_folder}")

        if not os.path.exists(upload_folder):
            print(f"Creando carpeta en: {upload_folder}")
            os.makedirs(upload_folder)

        filename = 'original_video.mp4'
        filepath = os.path.join(upload_folder, filename)
        print(f"Ruta completa de carga: {filepath}")

        file.save(filepath)

        return render_template('index.html', message=f'Video uploaded successfully! Everything is set to run the inference!')
    else:
        return render_template('index.html', message='File type not allowed')
    
@app.route('/run_inference', methods=['POST'])
def run_inference():
    stop_event.clear()
    thread = Thread(target=run_inference_thread)
    thread.start()
    return render_template('index.html', message=None, inference_message='Inference started...')

def run_inference_thread():
    global progress, inference_process
    stop_requested = False
    try:
        stop_requested = False
        progress["status"] = "running"
        progress["percent"] = 0
        progress["message"] = ""
        progress["start_time"] = time.time()

        import re

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        transunet_path = os.path.join(project_root, 'TransUNet')
        script_path = os.path.join(transunet_path, 'test.py')
        command = ['python3.8', '-u', script_path, '--dataset', 'Synapse', '--vit_name', 'R50-ViT-B_16']

        inference_process = subprocess.Popen(command, cwd=transunet_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in inference_process.stdout:
            if stop_event.is_set():
                inference_process.terminate()
                progress['status'] = 'Stopped'
                progress['message'] = '🛑 Inference stopped by user'
                return
            
            clean_line = line.strip()
            match = re.search(r'\[PROGRESS (\d+(?:\.\d+)?)\](.*)', clean_line)
            if match:
                progress["percent"] = float(match.group(1))
                progress["message"] = match.group(2).strip()
                progress["elapsed_time"] = int(time.time() - progress["start_time"])
                print(f"[PROGRESS] {progress['percent']}% - {progress['message']}")

        inference_process.wait()

        if not stop_event.is_set():
            if inference_process.returncode == 0:
                progress["percent"] = 100
                progress["message"] = "✅ Inference completed successfully!"
                progress["status"] = "done"
            else:
                progress["message"] = "❌ Error during inference"
                progress["status"] = "error"

    except Exception as e:
        progress["last_line"] = f"❌ Exception: {str(e)}"
        progress["status"] = "error"

@app.route('/stop_inference', methods=['POST'])
def stop_inference():
    global inference_process, progress
    stop_event.set()
    if inference_process and inference_process.poll() is None:
        inference_process.terminate()
        inference_process = None
        progress['status'] = 'stopped'
        progress['message'] = '🛑 Inference manually stopped'
    return '', 204

@app.route('/progress')
def get_progress():
    elapsed_time = 0
    if progress["status"] == "running" and "start_time" in progress:
        elapsed_time = int(time.time() - progress["start_time"])
    return jsonify({
        "status": progress["status"],
        "percent": progress["percent"],
        "message": progress["message"],
        "elapsed_time": elapsed_time
    })   

def allowed_file(filename):
    allowed_extensions = {'mp4'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/check_frames')
def check_frames():
    frames_dir = os.path.join('static', 'frames')
    # Verificar si el archivo frames_ready.flag está presente en el directorio
    flag_file_path = os.path.join(frames_dir, 'frames_ready.flag')
    
    if os.path.exists(flag_file_path):
        frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')]
        frames_available = len(frames) > 0  # Comprobar si hay frames disponibles
        return jsonify({'frames_available': frames_available})
    else:
        return jsonify({'frames_available': False})

@app.route('/get_frames')
def get_frames():
    frames_dir = os.path.join('static', 'frames')
    try:
        frames = sorted([
            f for f in os.listdir(frames_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        return jsonify({'frames': frames})
    except Exception as e:
        print(f"Error loading frames: {e}")
        return jsonify({'frames': []})
    
@app.route('/frames/<filename>')
def get_frame(filename):
    return send_from_directory(os.path.join('static', 'frames'), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# Directory where images are saved
IMAGE_DIR = '/home/eireland/ty/runs/obb/predict'

@app.route('/')
def index():
    # List all images in the directory
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    # Generate HTML to display the images
    html = '''
    <h1>Object Detection Results</h1>
    <meta http-equiv="refresh" content="5">  <!-- Refresh every 5 seconds -->
    '''
    for image in images:
        html += f'<div><img src="/images/{image}" width="640"></div>'
    return html

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

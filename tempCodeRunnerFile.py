import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# 학습된 모델 로드
building_model = load_model('model.h5')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# 이미지 전처리 함수
def preprocess_image(image):
    # 밝기 및 대비 조정
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
   
    # 히스토그램 평활화
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    hist_equalized = cv2.equalizeHist(gray)
    
    # 채널 병합
    preprocessed_image = cv2.merge([hist_equalized, hist_equalized, hist_equalized])
    return preprocessed_image

# 이미지에 번호판을 인식하고 해당 영역을 블러 처리하는 함수
def blur_license_plate(image):
    preprocessed_image = preprocess_image(image)  # 전처리된 이미지 사용
    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in plates:
        # 번호판 영역 확장
        expand_x = int(w * 0.3)
        expand_y = int(h * 0.3)
        x = max(0, x - expand_x)
        y = max(0, y - expand_y)
        w = min(image.shape[1], w + 2 * expand_x)
        h = min(image.shape[0], h + 2 * expand_y)
        plate_region = image[y:y+h, x:x+w]
        blurred_plate = cv2.GaussianBlur(plate_region, (51, 51), 0)
        image[y:y+h, x:x+w] = blurred_plate

    return image

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('select_file', filename=file.filename))
    return render_template('upload.html')

@app.route('/select/<filename>', methods=['GET', 'POST'])
def select_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # 원본 이미지 로드
    image = cv2.imread(file_path)

    # 번호판 인식 및 블러 처리
    blurred_image = blur_license_plate(image)

    # 블러 처리된 이미지 저장
    processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], 'blurred_' + filename)
    cv2.imwrite(processed_file_path, blurred_image)

    return render_template('show.html', original_image=filename, processed_image='blurred_' + filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
#http://127.0.0.1:5000/  이걸로 실행
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 학습된 모델 로드
model = tf.keras.models.load_model('model.h5')

# 예측 함수
def predict_building(image):
    # 이미지를 모델 입력 크기에 맞게 조정
    img = cv2.resize(image, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 예측 수행
    prediction = model.predict(img)
    print(f"Prediction: {prediction[0][0]}")  # 예측 결과 출력
    return prediction[0][0]  # 예측 결과 반환

# 이미지를 전처리하고 모델에 넣어 건물 여부 예측
def process_and_blur_image(image_path, model, blur_intensity=15):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open or find the image:", image_path)
    
    # 이미지 크기 가져오기
    h, w, _ = image.shape
    
    # 예측 수행
    prediction = predict_building(image)
    
    # 블러링할 영역 마스크 초기화
    mask = np.ones((h, w), dtype=np.uint8) * 255  # 흰색 마스크 (모두 블러 처리)

    # 임의의 사각형 영역 선택 (여기서는 중앙 영역)
    x_start = w // 4
    y_start = h // 4
    x_end = 3 * w // 4
    y_end = 3 * h // 4
    
    # 예측 결과에 따라 블러링할 영역 설정
    if prediction > 0.5:  # 0.5는 건물 여부의 임계값 (필요에 따라 조정)
        print("Building detected, applying blur to central area")
        mask[y_start:y_end, x_start:x_end] = 0  # 중앙 영역 블러 처리 안 함
    else:
        print("No building detected, blur applied to entire image")
        mask[:] = 0  # 전체 블러 처리 안 함

    # 블러링 적용
    blurred_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)
    
    # 마스크를 사용하여 원본 이미지와 블러 처리된 이미지를 결합
    final_image = np.where(mask[:, :, None] == 0, image, blurred_image)
    
    return final_image

# 블러 처리된 이미지 저장
def save_blurred_image(image, save_path):
    cv2.imwrite(save_path, image)

# 결과를 화면에 출력
def show_image(image, title='Image'):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 이미지 경로
image_path = '70.jpg'
save_path = '70_blurred_image.jpg'

# 이미지 처리 및 블러링
blurred_image = process_and_blur_image(image_path, model)

# 결과 저장
save_blurred_image(blurred_image, save_path)

# 결과 이미지 출력
cv2.namedWindow('Blurred License Plate Image', cv2.WINDOW_NORMAL)  # 이미지 창의 크기를 조절할 수 있는 창으로 설정
cv2.imshow('Blurred License Plate Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
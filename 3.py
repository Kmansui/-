import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# 학습된 모델 로드
building_model = load_model('model.h5')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# # 이미지에 건물을 인식하고 해당 영역을 블러 처리하는 함수
# def blur_building(image):
#     # 예측 함수
#     def predict_building(image):
#         img = cv2.resize(image, (150, 150))
#         img = img.astype('float32') / 255.0
#         img = np.expand_dims(img, axis=0)
#         prediction = building_model.predict(img)
#         return prediction[0][0] > 0.5  # 건물이면 True, 아니면 False
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     prediction = predict_building(image)
    
#     if prediction:
#         blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
#     else:
#         blurred_image = image
    
#     return blurred_image

# 이미지에 번호판을 인식하고 해당 영역을 블러 처리하는 함수
def blur_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    
    if len(plates) == 0:
        return image
    
    for (x, y, w, h) in plates:
        print(f"License plate found at: x={x}, y={y}, w={w}, h={h}")
        plate_region = image[y:y+h, x:x+w]
        blurred_plate = cv2.GaussianBlur(plate_region, (15, 15), 0)
        image[y:y+h, x:x+w] = blurred_plate

    return image

# 이미지 경로
image_path = '70.jpg'

# 이미지 로드
image = cv2.imread(image_path)

# 번호판 인식 및 블러 처리
blurred_image = blur_license_plate(image)

# # 건물 인식 및 블러 처리
# blurred_image = blur_building(blurred_image)

# 결과 이미지 출력
cv2.namedWindow('Blurred License Plate Image', cv2.WINDOW_NORMAL)  # 이미지 창의 크기를 조절할 수 있는 창으로 설정
cv2.imshow('Blurred License Plate Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
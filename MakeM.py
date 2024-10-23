import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import cv2
import numpy as np

# 데이터셋 경로 설정
train_dir = r'C:\Users\ckacl\OneDrive\바탕 화면\ppp\datase\자동차 차종-연식-번호판 인식용 영상\Training'
validation_dir = r'C:\Users\ckacl\OneDrive\바탕 화면\ppp\datase\자동차 차종-연식-번호판 인식용 영상\Validation'
# 이미지 데이터 생성기 설정
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 데이터 불러오기
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# CNN 모델 생성
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 한 에포크당 스텝 수
      epochs=30,
      validation_data=validation_generator,  # 검증 데이터 생성자 전달
      validation_steps=50  # 검증 데이터의 스텝 수
)

# 학습된 모델 저장
model.save('model.h5')

# # 학습된 모델 로드
# model = tf.keras.models.load_model('model.h5')

# # 예측 함수
# def predict_building(image):
#     # 이미지를 모델 입력 크기에 맞게 조정
#     img = cv2.resize(image, (150, 150))
#     img = img.astype('float32') / 255.0
#     img = np.expand_dims(img, axis=0)
    
#     # 예측 수행
#     prediction = model.predict(img)
#     return prediction[0][0]  # 예측 결과 반환

# # 이미지를 전처리하고 모델에 넣어 건물 여부 예측
# def process_and_blur_image(image_path, model, blur_intensity=15):
#     # 이미지 로드
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Could not open or find the image:", image_path)
    
#     # 이미지 크기 가져오기
#     h, w, _ = image.shape
    
#     # 예측 수행
#     prediction = predict_building(image)
    
#     # 블러링할 영역 마스크 초기화
#     mask = np.ones((h, w), dtype=np.uint8) * 255  # 흰색 마스크 (모두 블러 처리)

#     # 예측 결과에 따라 블러링하지 않을 영역 설정 (여기서는 간단히 전체 이미지 예측)
#     if prediction > 0.5:  # 0.5는 건물 여부의 임계값 (필요에 따라 조정)
#         mask[:] = 0  # 검은색 마스크 (블러 처리 안 함)

#     # 블러링 적용
#     blurred_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)
    
#     # 마스크를 사용하여 원본 이미지와 블러 처리된 이미지를 결합
#     final_image = np.where(mask[:, :, None] == 0, image, blurred_image)
    
#     return final_image

# # 블러 처리된 이미지 저장
# def save_blurred_image(image, save_path):
#     cv2.imwrite(save_path, image)

# # 이미지 경로
# image_path = '123.jpg'
# save_path = '123_blurred_image.jpg'

# # 이미지 처리 및 블러링
# blurred_image = process_and_blur_image(image_path, model)

# # 결과 저장
# save_blurred_image(blurred_image, save_path)
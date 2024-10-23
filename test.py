import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# 데이터셋 경로 설정
train_dir = r'C:\Users\ckacl\OneDrive\바탕 화면\ppp\dataset2\images\train'
validation_dir = r'C:\Users\ckacl\OneDrive\바탕 화면\ppp\dataset2\labels\val'
# 이미지 데이터 생성기 설정
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 데이터 불러오기
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')  # class_mode를 categorical로 변경

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')  # class_mode를 categorical로 변경

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
    layers.Dense(3, activation='softmax')  # 출력 뉴런 수를 클래스 수에 맞게 조정
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # loss 함수 변경
              metrics=['accuracy'])

# 모델 학습
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 한 에포크당 스텝 수
      epochs=30,
      validation_data=validation_generator,  # 검증 데이터 생성자 전달
      validation_steps=50)  # 검증 데이터의 스텝 수

# 학습된 모델 저장
model.save('model.h5')

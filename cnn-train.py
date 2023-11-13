import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Definir rutas de directorios de datos
train_dir = './DATA/TRAIN'
val_dir = './DATA/VAL'
test_dir = './DATA/TEST'

# Dimensiones de las imágenes
img_width, img_height = 224, 224

# Configurar el preprocesamiento de imágenes y generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# Definir modelo VGG16 pre-entrenado sin la capa densa superior
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Agregar capas densas superiores personalizadas
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

# Unir el modelo base y las capas densas personalizadas
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar los pesos del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Definir callbacks
early_stop = EarlyStopping(patience=5, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Entrenar modelo
history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stop, checkpoint])

# Evaluar modelo con conjunto de prueba
model.load_weights('best_model.h5')
test_loss, test_acc = model.evaluate_generator(test_generator)
print('Test accuracy:', test_acc)
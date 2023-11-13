from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

def classify_image(image_path):
    model = load_model('best_model.h5')
    img = load_img(image_path, target_size=(img_width, img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    prediction = model.predict(x)[0]

    classes = ['Normal', 'Neumonia_Bacteria', 'Neumonia_Virus']
    for i in range(len(classes)):
        print(f'{classes[i]}: {prediction[i] * 100:.2f}%')

    predicted_class = np.argmax(prediction)

    return classes[predicted_class]

print('TEST BACTERIA')
classify_image('./DATA/TEST/Neumonia_Bacteria/person126_bacteria_600.jpeg')

print('##############################################################################################')

print('TEST VIRUS')
classify_image('./DATA/TEST/Neumonia_Virus/person1609_virus_2790.jpeg')

print('##############################################################################################')

print('TEST NORMAL')
classify_image('./DATA/TEST/Normal/NORMAL2-IM-1427-0001.jpeg')
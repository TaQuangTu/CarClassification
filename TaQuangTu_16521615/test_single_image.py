import numpy as np
import keras.models
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array

from DataUtils.DataPreprocessing import get_class_names

model_path = "/content/drive/My Drive/Colab Notebooks/CarRecognition/Car-Recognition-master/MyOwn/Models/ResNet.69-0.87.hdf5"

model = keras.models.load_model(model_path)

test_image_path = "/content/drive/My Drive/Colab Notebooks/CarRecognition/Car-Recognition-master/data/lambogini.jpg"

image = load_img(test_image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, 0)
image = imagenet_utils.preprocess_input(image)

predict = model.predict(image)
max_index = np.argmax(predict)

print(predict)
print(max_index)
class_names= get_class_names('../devkit/cars_meta.mat')
print(class_names[max_index])
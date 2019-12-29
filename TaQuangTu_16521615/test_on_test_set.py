import keras
import numpy
import scipy.io
import os.path

from ImageUtils.ImageUtils import get_image_paths, read_image
from sklearn.metrics import confusion_matrix, classification_report

from ImageUtils.ImageUtils import read_multi_image

model_path = '/content/drive/My Drive/Colab Notebooks/CarRecognition/Car-Recognition-master/MyOwn/Models/ResNet_from_scratch.57-0.32.hdf5'
model = keras.models.load_model(model_path)
test_mat = scipy.io.loadmat('../devkit/cars_test_annos_withlabels.mat')

y_ground_truth = test_mat['annotations'][0]
y_rounded=[]
for result in y_ground_truth:
    car_class = result[4][0][0] - 1
    y_rounded.append(car_class)
y_rounded = numpy.asarray(y_rounded)

test_image_paths =  get_image_paths('/content/drive/My Drive/Colab Notebooks/CarRecognition/Car-Recognition-master/data/test',False)
test_image_paths.sort() # keep origin order
images = read_multi_image(test_image_paths)

y_predict = model.predict(images)
y_predict = numpy.argmax(y_predict,axis=1)

print(confusion_matrix(y_rounded, y_predict))
print(classification_report(y_rounded, y_predict))


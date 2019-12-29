import keras.models
import numpy as np
from ImageUtils import ImageUtils
from DataUtils import DataPreprocessing
from sklearn.metrics import classification_report,confusion_matrix
 
model = keras.models.load_model('/content/drive/My Drive/Colab Notebooks/CarRecognition/Car-Recognition-master/MyOwn/Models/ResNet.09-0.83.hdf5')
#vgg_model.summary()
test_image_paths = ImageUtils.get_image_paths('../data/train', do_shuffle=True)
print("reading test set===================")
x_test = ImageUtils.read_multi_image(test_image_paths)
y_test = DataPreprocessing.get_one_vs_hot_labels(test_image_paths)
# print(test_image_paths[0:20])
# print(np.argmax(y_test[0:20],axis=1) )
y_pred = model.predict(x_test,verbose=0)
# print(np.argmax(y_pred[0:20],axis=1) )

rounded_labels=np.argmax(y_test, axis=1)
rounded_pred=np.argmax(y_pred, axis=1)
print(confusion_matrix(rounded_labels,rounded_pred))
print(classification_report(rounded_labels,rounded_pred))

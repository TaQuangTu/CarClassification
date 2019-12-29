import keras
from keras import Model
from keras.layers import AveragePooling2D, Flatten, Dense
from keras.optimizers import SGD

def loadResNet50(weight,include_top, num_of_classes):
    model = keras.applications.resnet50.ResNet50(weights=weight, include_top=include_top, input_shape=(224, 224, 3))
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(model.output)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(num_of_classes, activation='softmax', name='fc8')(x_fc)

    model = Model(model.input, x_fc)
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def loadVGG16(weight,include_top, num_of_classes):
    model = keras.applications.vgg16.VGG16(weights=weight, include_top=include_top, input_shape=(224, 224, 3))
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(model.output)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(num_of_classes, activation='softmax', name='fc8')(x_fc)

    model = Model(model.input, x_fc)
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
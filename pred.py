import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def resNet(path):
    bmodel = tf.keras.applications.ResNet50V2(input_shape=(160,160,3),
                                                include_top=False)
    bmodel.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = bmodel(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) # Average pooling operation
    x = tf.keras.layers.BatchNormalization()(x) # Introduce batch norm
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    model.load_weights(path)
    return model

def efficientNet(path):

    bmodel = tf.keras.applications.EfficientNetV2B1(input_shape=(160,160,3),
                                                include_top=False)
    bmodel.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = bmodel(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) # Average pooling operation
    x = tf.keras.layers.BatchNormalization()(x) # Introduce batch norm
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    model.load_weights(path)
    return model

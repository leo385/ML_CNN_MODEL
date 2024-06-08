from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



# Krok 1. Wstępne przetwarzanie danych.
def initDataGenerator():
    global trainDataGen
    global validationDataGen

    trainDataGen = ImageDataGenerator(rescale=1.0/255)
    validationDataGen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')

def readImagesFromDirectory():
    if trainDataGen is not None and validationDataGen is not None:
        global train_generator
        global validation_generator

        train_generator = trainDataGen.flow_from_directory("/images/train", target_size=(200, 200), batch_size=10, class_mode='binary')
        validation_generator = validationDataGen.flow_from_directory("/images/validation", target_size=(150, 150), batch_size=32, class_mode='binary')


if __name__ == '__main__':
    initDataGenerator()
    readImagesFromDirectory()

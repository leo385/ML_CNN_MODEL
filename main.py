from keras import Model
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Sequential

from keras._tf_keras.keras.applications import VGG16


import matplotlib.pyplot as plt


# Krok 1. Wstępne przetwarzanie danych.
def initDataGenerator():
    global trainDataGen
    global validationDataGen

    trainDataGen = ImageDataGenerator(rescale=1.0 / 255)#,  # Normalizacja wartości pikseli
                                      #rotation_range=40,  # Losowe obracanie obrazu w zakresie 40 stopni
                                      #width_shift_range=0.2,  # Przesunięcie poziome obrazu o 20%
                                      #height_shift_range=0.2,  # Przesunięcie pionowe obrazu o 20%
                                      #shear_range=0.2,  # Pochylenie obrazu
                                      #zoom_range=0.2,  # Losowe przybliżenie obrazu
                                      #horizontal_flip=True,  # Losowe odbicie lustrzane w poziomie
                                      #fill_mode='nearest')  # Wypełnienie pustych miejsc najbliższymi wartościami)

    validationDataGen = ImageDataGenerator(rescale=1. / 255)


def readImagesFromDirectory():
    if trainDataGen is not None and validationDataGen is not None:
        global train_generator
        global validation_generator

        train_generator = trainDataGen.flow_from_directory("images/train/", target_size=(224, 224), batch_size=20,
                                                           class_mode='binary')
        validation_generator = validationDataGen.flow_from_directory("images/validation/", target_size=(224, 224),
                                                                     batch_size=20, class_mode='binary')


# Funkcja pozwalajaca trenowac model
def make_convnet(x, y):
    input_shape = (x, y, 3)

    model = Sequential([

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')

    ])

    return model


# Krok 2. Trenowanie modelu CNN
def initTrainingModels():
    model = make_convnet(200, 200)
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.fit(train_generator, steps_per_epoch=10, epochs=10)

    global history

    # trenowanie modelu z rozszerzonymi danymi
    history = model.fit(

        train_generator,
        steps_per_epoch=160,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=58

    )

def drawGraph():
    if history is not None:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label='Accuracy of training data')
        plt.plot(epochs, val_acc, 'b', label='Accuracy of validation data')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'r', label='Loss of training data')
        plt.plot(epochs, val_loss, 'b', label='Loss of validation data')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
    else:
        print("No training history available.")



# VGG-16 Model init

def VGG_InitBaseModel():

    global base_model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    global predictions
    predictions = Dense(1, activation='sigmoid')(x)

def VGG_TrainingModel():

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Trenuj model
    global history
    history = model.fit(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)


if __name__ == '__main__':

    # My CNN Model init
    initDataGenerator()
    readImagesFromDirectory()
    #initTrainingModels()


    # VGG-16 Model init
    VGG_InitBaseModel()
    VGG_TrainingModel()

    drawGraph()

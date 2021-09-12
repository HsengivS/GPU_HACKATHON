# For training set only
import glob
from logging import warning
import numpy as np
from numpy.core.fromnumeric import mean
import tensorflow as tf
from tensorflow.python.training.tracking.util import Checkpoint
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

non_text = glob.glob(r'E:\__tutorials\hackathon\gpu_hackathon\dataset\TRAIN\background\*')
text = glob.glob(r'E:\__tutorials\hackathon\gpu_hackathon\dataset\TRAIN\hi\*')

data = []
labels = []
for i in non_text:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(0)

for i in text:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(1)


train_data = np.array(data)
train_labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state = 42)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train /= 255
X_test /= 255

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, utils 

lb = LabelEncoder()
y_train = utils.to_categorical(lb.fit_transform(y_train))
y_test = utils.to_categorical(lb.fit_transform(y_test))

from tensorflow.keras.applications import VGG16
vgg_model = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

for layer in vgg_model.layers:
    layer.trainable = False

# Make sure you have frozen the correct layers
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)

from tensorflow.keras import layers, models, optimizers

x = vgg_model.output
x = layers.Flatten()(x) # Flatten dimensions to for use in FC layers
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(1, activation='softmax')(x) # Softmax for multiclass
transfer_model = models.Model(inputs=vgg_model.input, outputs=x)

learning_rate= 0.0001
transfer_model.compile(loss="categorical_crossentropy", optimizer = optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
history = transfer_model.fit(X_train, y_train, batch_size = 16, epochs=5, validation_data=(X_test,y_test))


#----------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
import warnings

warnings.filterwarnings("ignore")

vgg_model = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

for layer in vgg_model.layers:
    layer.trainable = False


x = layers.Flatten()(vgg_model.output)
#x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = layers.Dense(1, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = models.Model(inputs = vgg_model.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])

model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 224
img_width = 224

test_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True) # set validation split

test_generator = test_datagen.flow_from_directory(
    r"E:\__tutorials\hackathon\gpu_hackathon\dataset\TEST",
    target_size=(img_height, img_width),
    class_mode='binary',
    subset='training') # set as training data

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    validation_split=0.1) # set validation split

train_generator = train_datagen.flow_from_directory(
    r"E:\__tutorials\hackathon\gpu_hackathon\dataset\TRAIN",
    target_size=(img_height, img_width),
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    r"E:\__tutorials\hackathon\gpu_hackathon\dataset\TRAIN", # same directory as training data
    target_size=(img_height, img_width),
    class_mode='binary',
    subset='validation') # set as validation data

train_generator.class_indices

training_images = 5288
validation_images = 587

steps_per_epoch = training_images//32
validation_steps = validation_images//32

checkpoint = tf.keras.callbacks.ModelCheckpoint("model/vgg_16_2.hdf5", monitor="val_loss", verbose=1, save_best_only= True, save_weights_only=False, 
                                                mode="auto", save_freq="epoch", options=None)

early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0, patience=40, verbose=1, mode="auto")                                                

history = model.fit_generator(train_generator,
                   steps_per_epoch = steps_per_epoch,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results. 
                   epochs = 50,  # change this for better results
                   validation_data = validation_generator,
                   validation_steps = validation_steps,
                   callbacks=[checkpoint, early])

np.mean(history.history['accuracy'])
np.mean(history.history['val_accuracy'])

model.predict(train_generator)

dir(train_generator)


import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# downloads the dataset
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# parameters for loading dataset
batch_size = 32
image_height = 180
image_width = 180

# loads the downloaded dataset with parameters (80% for training)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

# validation dataset (20% for validation)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

# gets the class names from the dataset
class_names = train_dataset.class_names
print(class_names)


# # shows images from the dataset
# plt.figure(figsize=(10,10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):                                  # gets 9 images
#         ax = plt.subplot(3, 3, i + 1)                   # plots 3x3 grid
#         plt.imshow(images[i].numpy().astype("uint8"))   # actually shows image
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

# # check the output tensor size
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)          # (32, 180, 180, 3)
#   print(labels_batch.shape)         # (32, )
#   break

# buffered prefetching so you can yield data from disk without I/O blocking
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# standardize RGB (you want your input values to be small)
normalization_layer = layers.Rescaling(1. / 255.0)

# number of classes
num_classes = len(class_names)

model = tf.keras.Sequential()

model.add(layers.Rescaling(1. / 255.0, input_shape=(image_height, image_width, 3)))     # rescaling layer to make RGB values smaller
model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))                      # convolutional layer with 16 output channels, Recitfied Linear Activation function (outputs input if it is positive, otherwise 0) (it decides if a nueron should be activated or not)
model.add(layers.MaxPooling2D())                                                        # downsamples, lowers resolution, so model can better adapt to image shifts
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())                                                             # converts output from 3D tensor to 1D
model.add(layers.Dense(128, activation='relu'))                                         # 
model.add(layers.Dense(num_classes))                                                    # final output layer


# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training epoch #
epochs=10

# trains the model
history = model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=epochs
)
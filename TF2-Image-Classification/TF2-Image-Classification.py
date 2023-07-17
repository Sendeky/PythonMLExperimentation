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


plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):                                  # gets 9 images
        ax = plt.subplot(3, 3, i + 1)                   # plots 3x3 grid
        plt.imshow(images[i].numpy().astype("uint8"))   # actually shows image
        plt.title(class_names[labels[i]])
        plt.axis("off")


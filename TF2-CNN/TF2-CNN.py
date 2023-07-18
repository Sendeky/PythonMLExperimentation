import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# class names (dataset classes)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# # shows the images
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# A Convolutional Nueral Network takes tensors of shape (image_height, image_width, color_channels)    color_channels is (R,G,B)
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))         # A convolutional 2D layer: 32 output channels, 3x3 "Scanning Matrix",
                                                                                        # Recitfied Linear Activation function (outputs input if it is positive, otherwise 0),
                                                                                        # and input of 32x32 image with 3 color channels (R,G,B)
model.add(layers.MaxPooling2D((2,2)))                                                   # MaxPooling downsamples, makes lower resolution so that model can better adapt to features shifting in an image
                                                                                        # This is because convolutional layers record precise postion of feature in image. Makes them sensitive to shifting
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))                                  # Generally, as width & height shrink, you can afford more output channels

# model.summary()

model.add(layers.Flatten())                                    # unrolls 3D tensor to 1D
model.add(layers.Dense(64, activation='relu'))                 # final Conv2D has 64 channels, therefore output tensor has 64 channels
model.add(layers.Dense(10))                                    # output layer (has to equal to number of classes)

# model.summary()
""" Model Parameters
Total params: 122570 (478.79 KB)
Trainable params: 122570 (478.79 KB)
Non-trainable params: 0 (0.00 Byte)
"""

model.compile(optimizer='adam',                                                         # actually compiles the model.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),     # 
              metrics=['accuracy'])                                                     # tracks accuracy

history = model.fit(train_images, train_labels, epochs=10,              # fits model to train_data and validation_data
                    validation_data=(test_images, test_labels))


# # plots accuracy of validation
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.0, 1])
# plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)          # gets model loss and accuracy (in relation to test_dataset)

print("test_loss: ", test_loss)
print("test_accuracy: ", test_acc)
# loss of ~0.91, accuracy of ~0.7
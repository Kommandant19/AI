# Helper libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from keras.metrics import BinaryAccuracy, Precision, Recall

# Python source files
import capture_images


data_directory = 'data'


img_height = 256
img_width  = 256


labels = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
          'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]


# Load data
train = tf.keras.utils.image_dataset_from_directory(
  data_directory,
  validation_split=0.7,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=32
)

val = tf.keras.utils.image_dataset_from_directory(
  data_directory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=32
)


class_names = train.class_names
print(class_names)


# Visualize the data
plt.figure(figsize=(9, 9))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# Standardize the data
normalized_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_data  = train.map(lambda x, y: (normalized_layer(x), y))


class_quantity = len(class_names)


# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(class_quantity)
])


# Compile the model
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
)


# Train the model
epochs = 40
history = model.fit(
    train,
    validation_data=val,
    epochs=epochs
)

# Save the model
model.save('model.h5')
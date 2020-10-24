import numpy as np
import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def main():
  train_images = mnist.train_images()
  train_labels = mnist.train_labels()

  test_images = mnist.test_images()
  test_labels = mnist.test_labels()

  # normalize images
  train_images = (train_images / 255) - 0.5
  test_images = (test_images / 255) - 0.5

  print(train_images.shape)

  # flatten images
  train_images = train_images.reshape((-1, 784))
  test_images = test_images.reshape((-1, 784))

  # nn model
  model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
  ])

  # model setup
  model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  # execute model training
  model.fit(
    train_images, 
    to_categorical(train_labels), 
    epochs=5,
    batch_size=32,
  )

  # evaluate the model
  print()
  print("start test")
  model.evaluate(
  test_images,
  to_categorical(test_labels)
)

main()
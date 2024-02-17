# This repository contains some of my work from a deep learning course on Umeå University!

The code is written in python as jupyter nootbooks to add comments.

- lab 1: CNNs and bayesian optimization <br> 
- lab 2: Transfer learning on CNNs <br> 
- lab 3: LSTM <br>
- lab 4: RNN <br>
- lab 5: Q-Learning <br><br>

#### If you have any questions regarding the work, please contact me

# Example of classification of the Fashion Mnist dataset

## Transfer learinng of the some of the structure/weights from Xception model


```Python
# Import needed libraries
import tensorflow as tf
print('TensorFlow version:', tf.__version__)

# from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.utils  import to_categorical

#print('Keras version:',tf.keras.__version__)

# Helper libraries
import numpy as np
import sklearn
from   sklearn.model_selection import train_test_split

# Matlab plotting
import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
earlystop= EarlyStopping(monitor='val_loss', patience=3, restore_best_weights = True)



# Get Fashion-MNIST training and test data from Keras database (https://keras.io/datasets/)
(train_images0, train_labels0), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Define labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Split the training set into a training and a validation set (20% is validation)
train_images, val_images, train_labels, val_labels = train_test_split(train_images0, train_labels0, test_size=0.20)

train_images = np.expand_dims(train_images, -1)
val_images = np.expand_dims(val_images, -1)
test_images = np.expand_dims(test_images, -1)

# Normalize the images.
train_images = (train_images - np.min(train_images)) / (np.max(train_images) - np.min(train_images))
test_images = (test_images - np.min(test_images)) / (np.max(test_images) - np.min(test_images))
val_images = (val_images - np.min(val_images)) / (np.max(val_images) - np.min(val_images))

# Define an in-stream transform (gray2color, resize)
def img_transform(images):
  images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images))
  images = tf.image.resize_with_pad(images, 32, 32, antialias=False)


  return images

# Ändrar dimensionerna på datamaterialen
train_images=img_transform(train_images)
val_images=img_transform(val_images)
test_images=img_transform(test_images)


# Ändrar så datat är rätt dimensioner
train_images=tf.image.resize_with_pad(train_images, 72, 72, antialias=False)
val_images=tf.image.resize_with_pad(val_images, 72, 72, antialias=False)
test_images=tf.image.resize_with_pad(test_images, 72, 72, antialias=False)

print(train_images.shape)
print(val_images.shape)
print(test_images.shape)

image_index = [42, 789] # "Random" images to print

for index in image_index:
  print( 'Label:', class_names[train_labels[index]])
  plt.figure()
  plt.imshow(train_images[index])
  plt.grid(False)
  plt.show(block=False)



input_shape = test_images[0].shape
# Xception från keras
Xception_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=input_shape,classes=10)

X_model = tf.keras.models.Model(inputs=Xception_model.inputs, outputs=Xception_model.get_layer('block5_sepconv1').output )

# Väljer ut output vid conv2D_3
inputs = Xception_model.inputs

X = Xception_model.get_layer('block5_sepconv1').output



# Lägger till egna lager som ska tränas
X= tf.keras.layers.BatchNormalization(axis=-1)(X) # normaliserar inputs
X= (Dropout(0.2))(X) # Dropout på 20% av noderna på faltningslagret som skapas nedan
X=(Conv2D(filters=96,
                        kernel_size=5,
                        activation='relu',
                        padding='same'))(X)

X=(tf.keras.layers.Flatten())(X) # Ser till så lagren är 'full connected'
# X_model.add(tf.keras.layers.Dense(294,activation='relu'))# Lägger in 2 gömda lager med olika antal noder
X=(tf.keras.layers.Dense(64,activation='relu'))(X)
outputs=(Dense(units=10, activation='softmax'))(X) #sista lagret


X_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
# Freeze the layers
for layer in X_model.layers[:-6]:
    layer.trainable = False

X_model.summary()

X_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss='categorical_crossentropy',
  metrics=['categorical_accuracy'],

)


epochs = 50 # Antal epoker som kan köras
batch_size = 96 # batchstoleken

# Train the model.
history = X_model.fit(
  train_images, to_categorical(train_labels),
  epochs=epochs,
  batch_size=batch_size,
  verbose = 1,
  validation_data=(val_images, to_categorical(val_labels)),
  callbacks=[earlystop] # Lägger in earlystopping i träningen
)

# Evaluate the model.
test_loss, test_acc = X_model.evaluate(test_images,to_categorical(test_labels))
print('Test accuracy: %.3f' % test_acc)

```
<br><br><br>


<div align="center">
  <img src="https://media1.tenor.com/m/ynucqTzAVkwAAAAC/machine-learning-normalizing-flows.gif" width="600" height="300"/>
</div>


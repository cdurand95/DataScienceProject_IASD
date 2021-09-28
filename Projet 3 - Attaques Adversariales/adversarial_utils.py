from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dropout,Dense, Activation, Input, Conv2D, Flatten, MaxPooling2D, BatchNormalization,AveragePooling2D
from tensorflow.keras import regularizers,optimizers, Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from keras.models import load_model
from tensorflow.keras import backend as K
from keras.layers import Lambda


def load_CIFAR10_data():
  ## import CIFAR-10
  (train_images, train_labels),(test_images, test_labels)= cifar10.load_data()

  ## nb class and shape of images
  nb_classes = 10
  y_train = to_categorical(train_labels,nb_classes)
  y_test = to_categorical(test_labels,nb_classes)

  ## Normalize
  x_train = train_images.astype("float32") / 255
  x_test = test_images.astype("float32") / 255

  return x_train,y_train,train_labels,x_test,y_test,test_labels,


def signed_gradient(model,test_images,y_test):
  images = tf.convert_to_tensor(test_images,dtype=tf.float32)
  with tf.GradientTape() as tape:
    tape.watch(images)
    predictions =model(images)
    loss = keras.losses.categorical_crossentropy(y_test, predictions)

  grad = tape.gradient(loss, images)
  signed_grad = tf.sign(grad)
  return signed_grad



def signed_grad_wrt_one_image(model,image,labels):
  with tf.GradientTape() as tape:
    tape.watch(image)
    prediction =model(image)
    loss = keras.losses.categorical_crossentropy(labels.reshape((1,10)), prediction)

  grad = tape.gradient(loss, image)
  signed_grad_one = tf.sign(grad)
  return signed_grad_one



def PGD_attack(model,data,iterations,eta,epsilon,labels):

  x = tf.identity(data)

  if data.shape[0] == 1 : # small difference in computing the loss for one image (reshaping)
    for i in range(iterations):
      signed_grad = signed_grad_wrt_one_image(model,x,labels)
      x =  x + eta * signed_grad
      x = np.clip(x,data-epsilon,data+epsilon)
      x = tf.convert_to_tensor(x,dtype=tf.float32)
  
  else :
    for i in range(iterations) :

      signed_grad = signed_gradient(model,x,labels)

      x =  x + eta * signed_grad
      x = np.clip(x,data-epsilon,data+epsilon)
      x = tf.convert_to_tensor(x,dtype=tf.float32)

  return x


def adversarial_set(model,train_images,y_train,percentage_of_adversarial,PGD_or_FGSM,epsilon,eta,iterations):
  train_shape = train_images.shape 
  x = np.zeros(train_shape)
  y = np.zeros(y_train.shape)
  index = 0
  x_adversarial = np.zeros(train_shape)

  if 0 <= percentage_of_adversarial <= 1 :
    index = int(np.floor(percentage_of_adversarial*train_shape[0]))
  else :
    raise ValueError("Specify a percentage of adversarial input between 0 and 1")

  if PGD_or_FGSM == "FGSM":
    signed_grad = signed_gradient(model,train_images,y_train)
    x_adversarial = train_images + epsilon*signed_grad
    x_adversarial = x_adversarial.numpy()
    
  elif PGD_or_FGSM == "PGD" :
    x_adversarial = PGD_attack(model,train_images,iterations,eta,epsilon,y_train).numpy()

  else :
    raise NameError('Specify if PGD or FGSM')
  
  if percentage_of_adversarial == 0 :
    return (train_images,y_train)
  elif percentage_of_adversarial == 1 :
    return (x_adversarial,y_train)
  else :
    adversarial_indexes = random.sample(range(train_shape[0]),index)
    normal_indexes = random.sample(range(train_shape[0]),train_shape[0]-index)

    x[:index] = x_adversarial[adversarial_indexes]
    y[:index] = y_train[adversarial_indexes]

    x[index:] = train_images[normal_indexes]
    y[index:] = y_train[normal_indexes]

    x, y = shuffle(x, y, random_state=0)

    return (x,y)



def categorical_accuracy(y_true, y_pred):
    return float(sum(K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx()))/len(y_true))



def loss(y_pred,y_true):
   return sum(keras.losses.categorical_crossentropy(y_true, y_pred))/len(y_true)



def print_perfdict_adversarial_training(perf_dict):
  plt.figure(figsize=(20,12))
  plt.subplot(221)
  plt.plot(perf_dict['loss'])
  plt.plot(perf_dict['original_output_loss'])
  plt.plot(perf_dict['adversarial_output_loss'])
  plt.title('Train set categorical crossentropy loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Weighted loss','Original loss','Adversarial loss'],loc = 'upper right')


  plt.subplot(222)
  plt.plot(perf_dict['val_loss'])
  plt.plot(perf_dict['val_original_output_loss'])
  plt.plot(perf_dict['val_adversarial_output_loss'])
  plt.title('Validation set categorical crossentropy loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Val weighted loss','Val original loss', 'Val Adversarial loss'], loc='upper right')

  plt.subplot(223)
  plt.plot(perf_dict['original_output_accuracy'])
  plt.plot(perf_dict['adversarial_output_accuracy'])
  plt.title('Train set categorical crossentropy accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Original accuracy', 'Adversarial accuracy'], loc='upper right')

  plt.subplot(224)
  plt.plot(perf_dict['val_original_output_accuracy'])
  plt.plot(perf_dict['val_adversarial_output_accuracy'])
  plt.title('Validation set categorical crossentropy accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Val original accuracy', 'Val Adversarial accuracy'], loc='upper right')
  plt.show()


## classification model for cifar10
def build_model_CNN_2(input_shape):
  input = Input(shape=input_shape)
  weight_decay = 1e-4
  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer= regularizers.l2(weight_decay), padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2),padding='same'))

  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer= regularizers.l2(weight_decay), padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2),padding='same'))
  model.add(Dropout(0.2))

  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer= regularizers.l2(weight_decay), padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2),padding='same'))
  
  model.add(AveragePooling2D((3, 3),padding='same'))
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer= regularizers.l2(weight_decay)))
  model.add(Dropout(0.2))
  model.add(Dense(10, activation='softmax'))

  return model


def ensemble_predictions(ensemble_models, Data):
	# make predictions
	yhats = [model.predict(Data) for model in ensemble_models]
	yhats = np.array(yhats)
	# sum across ensemble ensemble_models
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result

### Attacks for ensemble models

def ensemble_signed_gradient(ensemble_model,test_images,y_test):
  images = tf.convert_to_tensor(test_images,dtype=tf.float32)
  signed_grad = np.zeros(test_images.shape)
  n = len(ensemble_model)
  for model in ensemble_model :
    with tf.GradientTape() as tape:
      tape.watch(images)
      predictions =model(images)
      loss = keras.losses.categorical_crossentropy(y_test, predictions)

    grad = tape.gradient(loss, images)
    signed_grad = signed_grad + 1/n * tf.sign(grad)
  return signed_grad


def ensemble_signed_grad_wrt_one_image(ensemble_model,image,labels):
  signed_grad_one = np.zeros(image.shape)
  n = len(ensemble_model)
  for model in ensemble_model :
    with tf.GradientTape() as tape:
      tape.watch(image)
      prediction =model(image)
      loss = keras.losses.categorical_crossentropy(labels.reshape((1,10)), prediction)

    grad = tape.gradient(loss, image)
    signed_grad_one = signed_grad_one + 1/n* tf.sign(grad)
  return signed_grad_one


def ensemble_PGD_attack(ensemble_model,data,iterations,eta,epsilon,labels):

  x = tf.identity(data)

  if data.shape[0] == 1 : # small difference in computing the loss for one image (reshaping)
    for i in range(iterations):
      signed_grad = ensemble_signed_grad_wrt_one_image(ensemble_model,x,labels)
      x =  x + eta * signed_grad
      x = np.clip(x,data-epsilon,data+epsilon)
      x = tf.convert_to_tensor(x,dtype=tf.float32)
  
  else :
    for i in range(iterations) :

      signed_grad = ensemble_signed_gradient(ensemble_model,x,labels)

      x =  x + eta * signed_grad
      x = np.clip(x,data-epsilon,data+epsilon)
      x = tf.convert_to_tensor(x,dtype=tf.float32)

  return x
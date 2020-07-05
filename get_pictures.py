
import nibabel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import seaborn as sns
import keras
from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import os
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as tf_back

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(0)
random.seed(0)

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

from lab_original import PreProcessor,DataLoader,Autoencoder



def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
    if tf_back._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tf_back._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tf_back._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tf_back._get_available_gpus = _get_available_gpus

roots = ["E:\Lab\MICCAI_BraTS_2019_Data_Training\MICCAI_BraTS_2019_Data_Training/HGG/",
         "E:\Lab\MICCAI_BraTS_2019_Data_Training\MICCAI_BraTS_2019_Data_Training/LGG/"]

patient_folders = [os.path.join(roots[0], p) for p in os.listdir(roots[0])] + \
                  [os.path.join(roots[1], p) for p in os.listdir(roots[1])]

# data = DataLoader.download_data(patient_folders)


#data = np.load("./dataset.npy")
#train_data, vald_data, test_data = PreProcessor.split(data, save=True)

train_data, vald_data, test_data = np.load("./dataset_split.npy", allow_pickle= True)
model = Autoencoder()

# model.run(train_data, vald_data)

model.load("./Models/Best_Model_04_07_2.h5")

pred = model.autoencoder.predict(train_data[np.newaxis,0])

f1 = plt.figure(figsize=(12, 12))
ax1 = f1.add_subplot(221)
ax2 = f1.add_subplot(222)
ax3 = f1.add_subplot(223)
ax4 = f1.add_subplot(224)

ax1.imshow(train_data[1][0], cmap="gray")
ax2.imshow(train_data[1][1], cmap="gray")
ax3.imshow(train_data[1][2], cmap="gray")
ax4.imshow(train_data[1][3], cmap="gray")
plt.show()

f2 = plt.figure(figsize=(12, 12))
ax5 = f2.add_subplot(221)
ax6 = f2.add_subplot(222)
ax7 = f2.add_subplot(223)
ax8 = f2.add_subplot(224)


ax5.imshow(pred[0][0], cmap="gray")
ax6.imshow(pred[0][1], cmap="gray")
ax7.imshow(pred[0][2], cmap="gray")
ax8.imshow(pred[0][3], cmap="gray")
plt.show()

f3 = plt.figure(figsize=(12, 12))
ax11 = f3.add_subplot(221)
ax12 = f3.add_subplot(222)
ax13 = f3.add_subplot(223)
ax14 = f3.add_subplot(224)


ax11.imshow(pred[0][0]-train_data[1][0], cmap="gray")
ax12.imshow(pred[0][1]-train_data[1][1], cmap="gray")
ax13.imshow(pred[0][2]-train_data[1][2], cmap="gray")
ax14.imshow(pred[0][3]-train_data[1][3], cmap="gray")
plt.show()
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend
from keras import optimizers

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix

real = pd.read_csv('True.csv', header = None, skiprows = 1)
fake = pd.read_csv('Fake.csv', header = None, skiprows = 1)

all_data = pd.concat([real, fake])

l = all_data[0].isin(real[0])
l = l.astype(int)
all_data[4] = l

all_data = all_data.sample(frac=1).reset_index()

all_data = all_data.iloc[0:7500]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

title_matrix = embed(all_data[0].tolist())

content_matrix = embed(all_data[1].tolist())

train_data = all_data.loc[0:int(len(all_data)*0.8)]
test_data = all_data.loc[int(len(all_data)*0.8):len(all_data)]

pca = PCA(n_components=3)
pca_data = pca.fit(title_matrix[0:len(train_data)])
pca_train = pca.transform(title_matrix[0:len(train_data)])

pca_3d = pd.DataFrame({0 : pca_train[:,0], 1 : pca_train[:,1], 2 : pca_train[:,2], 3 : train_data[4]})

X_train_title = pca_3d[[0, 1, 2]]
y_train_title = pca_3d[3]

model_title = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(3)),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(16, activation='relu'),
                                    tf.keras.layers.Dense(4, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

model_title.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_title.fit(X_train_title, y_train_title, batch_size=1, epochs=25)

pca_content = PCA(n_components=3)
pca_data_content = pca.fit(content_matrix[0:len(train_data)])
pca_train_content = pca.transform(content_matrix[0:len(train_data)])

pca_3d_content = pd.DataFrame({0 : pca_train_content[:,0], 1 : pca_train_content[:,1], 2 : pca_train_content[:,2], 3 : train_data[4]})

X_train_content = pca_3d_content[[0, 1, 2]]
y_train_content = pca_3d_content[3]

model_content1 = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(3)),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(16, activation='relu'),
                                    tf.keras.layers.Dense(4, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

model_content1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_content2 = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(3)),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(16, activation='relu'),
                                    tf.keras.layers.Dense(4, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

model_content2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_content1.fit(X_train_content, y_train_content, batch_size=1, epochs=25)

model_content2.fit(X_train_content, y_train_content, batch_size=1, epochs=25)

def custom(data, article=True):
    if article==True:
        title_vector = embed(data[:data.index('\n')])
        content_vector = embed(data[data.index('\n')+2:])
        pca_title = pca.fit(title_vector)
        pca_title_format = pca.transform(title_vector)
        pca_title_3d = pd.DataFrame({0 : pca_title_format[:,0], 1 : pca_title_format[:,1], 2 : pca_title_format[:,2]})
        pca_content = pca.fit(content_vector)
        pca_content_format = pca.transform(content_vector)
        pca_content_3d = pd.DataFrame({0 : pca_content_format[:,0], 1 : pca_content_format[:,1], 2 : pca_content_format[:,2]})
        result = []
        result.append(model_title.predict(pca_title_3d))
        result.append(model_content1.predict(pca_content_3d))
        result.append(model_content2.predict(pca_content_3d))
        count = 0
        for i in range(len(result)):
            if result > 0.5:
                count+=1
        if count > 1:
            return model_content1.predict(pca_content_3d)
        else:
            return model_title.predict(pca_title_3d)
    else:
        pca_content = pca.fit(content_vector)
        pca_content_format = pca.transform(content_vector)
        pca_content_3d = pd.DataFrame({0 : pca_content_format[:,0], 1 : pca_content_format[:,1], 2 : pca_content_format[:,2]})
        return model_content1.predict(pca_content_3d)
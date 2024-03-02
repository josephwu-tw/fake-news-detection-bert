# setup

import __path__ as path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from transformers import AutoTokenizer

keras.backend.clear_session()

# functions

def show_train_history(train_history, train, validation, title = 'Train History'):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()

# data import

x_train_bert = pd.read_csv(path.data_dir + 'x_train_bert.csv')
y_train = pd.read_csv(path.data_dir + 'y_train.csv')
x_valid_bert = pd.read_csv(path.data_dir + 'x_valid_bert.csv')
y_valid = pd.read_csv(path.data_dir + 'y_valid.csv')

# set hyperparameter

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 512
vocab_size = tokenizer.vocab_size + 1

# data transform

x_train_bert = np.array(x_train_bert, dtype = float)
x_valid_bert = np.array(x_valid_bert, dtype = float)

y_train = np.array(y_train, dtype = float)
y_valid = np.array(y_valid, dtype = float)

# build model

Input_BERT = keras.Input(shape = (MAX_LEN,))
Embedding_Layer = layers.Embedding(vocab_size, 64, trainable = True)(Input_BERT)
LSTM_Layer = layers.LSTM(64, return_sequences = True, dropout = 0.3)(Embedding_Layer)
Bidir_Layer = layers.Bidirectional(layers.LSTM(64, dropout = 0.3))(LSTM_Layer)
Output = layers.Dense(1, activation = 'sigmoid')(Bidir_Layer) 

model_bert = keras.Model(inputs = Input_BERT, outputs = Output)
    
model_bert.summary()

# compile

model_bert.compile(loss = 'binary_crossentropy',
                    optimizer = Adam(learning_rate = 0.01),
                    metrics = ['accuracy'])

# train

history_bert = model_bert.fit(x_train_bert, y_train,
                                epochs = 10,
                                batch_size = (len(x_train_bert)//10),
                                validation_data = (x_valid_bert, y_valid))
    
show_train_history(history_bert, 'loss', 'val_loss', title = 'Train History - BERT')

# save model

model_bert.save(path.model_dir + 'bert.keras')


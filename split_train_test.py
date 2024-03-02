# setup

import __path__ as path
import pandas as pd
import numpy as np
import sklearn.preprocessing as skpp
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# data import

df = pd.read_csv(path.data_dir + 'clean_dataset.csv').dropna()

# tokenize

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 512
vocab_size = tokenizer.vocab_size + 1

df['text'] = df['text'].apply(lambda text: tokenizer.encode_plus(text, padding = 'max_length',
                                                                 max_length = MAX_LEN, truncation = True))

# scaling

scaler = skpp.MinMaxScaler(feature_range = (0,1))
df.iloc[:,10:] = scaler.fit_transform(df.iloc[:,10:])

# get train, test, and valid

X = []
Y = []

for i in range(len(df)):

    X.append([df.iloc[i,0], df.iloc[i,2:].values.tolist()])
    Y.append([df.iloc[i,1].astype(float)])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle = False)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, shuffle = False)

x_train_bert, x_train_mlp = [x[0]['input_ids'] for x in x_train], [x[1] for x in x_train]
x_valid_bert, x_valid_mlp = [x[0]['input_ids'] for x in x_valid], [x[1] for x in x_valid]
x_test_bert, x_test_mlp = [x[0]['input_ids'] for x in x_test], [x[1] for x in x_test]

# output train, test, valid

pd.DataFrame(x_train_bert).to_csv(path.data_dir + 'x_train_bert.csv', index = False)
pd.DataFrame(x_train_mlp).to_csv(path.data_dir + 'x_train_mlp.csv', index = False)
pd.DataFrame(y_train).to_csv(path.data_dir + 'y_train.csv', index = False)
pd.DataFrame(x_test_bert).to_csv(path.data_dir + 'x_test_bert.csv', index = False)
pd.DataFrame(x_test_mlp).to_csv(path.data_dir + 'x_test_mlp.csv', index = False)
pd.DataFrame(y_test).to_csv(path.data_dir + 'y_test.csv', index = False)
pd.DataFrame(x_valid_bert).to_csv(path.data_dir + 'x_valid_bert.csv', index = False)
pd.DataFrame(x_valid_mlp).to_csv(path.data_dir + 'x_valid_mlp.csv', index = False)
pd.DataFrame(y_valid).to_csv(path.data_dir + 'y_valid.csv', index = False)
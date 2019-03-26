import pandas as pd
df = pd.read_json("C:/Projects/Saecasm-Detection/Sarcasm_Headlines_Dataset.json", lines=True)
df = df.drop(['article_link'], axis=1)
df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))
print(df.head())

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential

max_features = 10000
maxlen = 25
embedding_size = 200

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['headline']))
X = tokenizer.texts_to_sequences(df['headline'])
X = pad_sequences(X, maxlen = maxlen)
y = df['is_sarcastic']
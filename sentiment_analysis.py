#sentiment analysis without pre processing done previously. this file is not tested on different examples because of lack of computational power. 
# but performed good in the testing cases i choosed.
import numpy as np 
from keras.layers import Dense  ,LSTM 
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
X=[]
Y=[]
num_words=5000
train_pos_dir="/home/hardik/Desktop/Sentiment_Analysis/imdb_data/train/pos/"
train_neg_dir="/home/hardik/Desktop/Sentiment_Analysis/imdb_data/train/neg/"

train_pos=[train_pos_dir+i for i in os.listdir(train_pos_dir)]
train_neg=[train_neg_dir+i for i in os.listdir(train_neg_dir)]

# extracting positive comments 
for text_pos in train_pos:
	with open(text_pos, 'r',encoding="utf-8") as file:
		text=file.read()
		X.append(text)
# saving positive as 1
	Y.append(1)

# extracting negative comments 
for text_neg in train_neg:
	with open(text_neg, 'r',encoding="utf-8") as file:
		text=file.read()
	X.append(text)
# saving negative as 0
	Y.append(0)

X=np.array(X)
Y=np.array(Y)
#X,Y=shuffle(X,Y)

tokenizer=Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n.br', lower=True)
tokenizer.fit_on_texts(X)
length_text=max(len(line.split()) for line in X)
vocab_size=len(tokenizer.word_counts)+1
print("[INFO]vocab size: ",vocab_size)
print("[INFO]Max Length of the  data:",length_text)

Xtrain=tokenizer.texts_to_sequences(X)
Xtrain=pad_sequences(Xtrain,maxlen=500,padding="pre")
validation_split = 1000 / len(Xtrain)
#print(X[2300])
#print(Xtrain[2300])

model=Sequential()
model.add(Embedding(num_words,32,input_length=500))
model.add(LSTM(200))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
print(model.summary())
model.fit(x=Xtrain,y=Y,batch_size=64,epochs=3,validation_split=validation_split,verbose=2)
model.save("/home/hardik/Desktop/Sentiment_Analysis/sentiment.h5")

text="good movie and fantastic cast and perfect story line."
test=tokenizer.texts_to_sequences(text)
pad_text=pad_sequences(test,maxlen=500,padding="pre")
print(model.predict(pad_text))
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
np_load_old = numpy.load

# modify the default parameters of np.load
numpy.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# using dataset available on the internet directly from keras . the dataset availble at keras is cleaned and does not require pre-processng.
# i tried using using raw dataset but my computer was not able to perform the task and kept showing memory error 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
numpy.load = np_load_old

# fixing the size of input sequence 
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
#checking the accuracy of the trained model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

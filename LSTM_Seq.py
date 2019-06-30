import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


class LSTMSEQ():
    """LSTM Network for sequence classification

    This method generates LSTM network for sequence classification

    Parameters
    ----------
    add_dp : Boolean (Default = False)
        Add dropout in the network to reduce overfitting.

    add_conv : Boolean (Default = False)
         Add convolutional layer to capture spatial structure in the sequence

    vocab_size : int (Default = 100)
         Size of vocabulary.

    seq_length : int (Default = 10)
         Length of the sequence.

    embedding_vecor_length : int (Default = 50)
          Length of the word embeddings.

    epochs : int (Default = 3)
          Number of Epochs.

    """



    def __init__(self, add_dp=False,
                 add_conv=False,
                 vocab_size=100,
                 seq_length=10,
                 embedding_vector_length=50,
                 epochs=3):
        self.add_dp = add_dp
        self.add_conv = add_conv
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_vector_length = embedding_vector_length
        self.epochs = epochs

    def fit(self, X, y, X_val, y_val):
        """Prepare the LSTM model to classify the sequence as 1 or 0.

        Parameters
        ----------
        X : array of shape = [n_samples, seq_length] with the data.

        y : class labels of each sample in X.

        Returns
        -------
        model
        """

        class_weights = class_weight.compute_class_weight('balanced',numpy.unique(y),y)
        #print("class weights", class_weights)
        #Create a LSTM model
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_vector_length, input_length=self.seq_length))
        if self.add_conv is True:
            model.add(Conv1D(filters=50, kernel_size=3, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))

        model.add(LSTM(50))
        if self.add_dp is True:
            model.add(Dropout(0.2))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=64, class_weight=class_weights)
        print(model.summary())
        return model


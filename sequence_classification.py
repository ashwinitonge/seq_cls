import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
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
from sklearn.externals import joblib

from LSTM_Seq import LSTMSEQ

seed = 7
numpy.random.seed(seed)

# load Train dataset
dataframe = pandas.read_csv("Sequence_model_data/train.csv")
dataset = dataframe.values

X = dataset[:,1:199].astype(float)
X = X[:,::-1]
y = dataset[:,0].astype(str)

# load Test dataset
dataframe_test = pandas.read_csv("Sequence_model_data/test.csv")
dataset_test = dataframe_test.values

X_test = dataset_test[:,0:199].astype(float)
X_test = X_test[:,::-1]


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

indices = numpy.arange(len(X))
numpy.random.shuffle(indices)
X = X[indices]
encoded_y = encoded_y[indices]

X_train, X_val, y_train, y_val = train_test_split(X, encoded_y, test_size=0.25, random_state=42)
print("Number of Features", X_train.shape[1])

print("Train/Test split created")

embedding_vector_length = 50
vocab_size = 653
seq_length = 198

#Create a LSTM model
lstmseq = LSTMSEQ()
lstmseq.vocab_size = vocab_size
lstmseq.seq_length = seq_length
lstmseq.embedding_vector_length = embedding_vector_length
lstmseq.epochs = 16
model_lstm = lstmseq.fit(X_train, y_train, X_val, y_val)

# Final evaluation of the model
preds = model_lstm.predict(X_val)

fpr, tpr, thresholds = metrics.roc_curve(y_val, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)

for i in range(0, len(preds)):
    preds[i] = 1 if preds[i] > 0.5 else 0

print('AUC performance, F1-score, Accuracy, TN, FP, FN, TP')
tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
print(auc, f1_score(y_val, preds), accuracy_score(y_val, preds), tn, fp, fn, tp)


#===============================================================================================================
#LSTM model with convolution and dropouts

lstmconvseq = LSTMSEQ()
lstmconvseq.add_dp = True
lstmconvseq.add_conv = True
lstmconvseq.vocab_size = vocab_size
lstmconvseq.seq_length = seq_length
lstmconvseq.embedding_vector_length = embedding_vector_length
lstmconvseq.epochs = 16
model_lstm_conv = lstmconvseq.fit(X_train, y_train, X_val, y_val)

# Final evaluation of the model
preds = model_lstm_conv.predict(X_val)

fpr, tpr, thresholds = metrics.roc_curve(y_val, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)

for i in range(0, len(preds)):
    preds[i] = 1 if preds[i] > 0.5 else 0

print('AUC performance, F1-score, Accuracy, TN, FP, FN, TP')
tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
print(auc, f1_score(y_val, preds), accuracy_score(y_val, preds), tn, fp, fn, tp)

#===============================================================================================================
#Save model

joblib.dump(model_lstm, "Sequence_model/lstm.pkl")
joblib.dump(model_lstm_conv, "Sequence_model/lstm_conv.pkl")

#===============================================================================================================
#Test predictions

pred_test_file = open("res/TestPredictions.csv", "w")

preds_test = model_lstm.predict(X_test)

for i in range(0, len(preds_test)):
    pred_test_file.write(str(preds_test[i]) + '\n')

pred_test_file.close()
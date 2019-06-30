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

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


def initVector(featureCount):
  if(featureCount <= 0):
    raise Exception('feature count should be greater than 0')

  return [0 for f in range(featureCount)]

def populateVector(featureVector, dataArray):
  if(len(featureVector) <= 0):
    raise Exception('featureVector should be non-empty')
  if(len(dataArray) <= 0):
    raise Exception('dataArray should be non-empty')

  #print(featureVector);
  for data in dataArray:
    #print(data)
    featureVector[data] += 1



seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("Sequence_model_data/train.csv")
dataset = dataframe.values
# split into input (X) and output (Y) variables
#X = dataset[1:-1,1:61].astype(float)
#Y = dataset[1:-1,0].astype(str)

X = dataset[:,1:199].astype(int)
y = dataset[:,0].astype(str)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)


#find the max value in the input
input = X
maxInput = numpy.amax(input)
maxInput +=1

print(maxInput)

#build a matrix for the data
inputVocabMatrix = []
vocab = range(0, maxInput)

for dataArray in input:
  featureVector = initVector(maxInput)
  populateVector(featureVector, dataArray)
  inputVocabMatrix.append(featureVector)


res = dict(zip(vocab,
               mutual_info_classif(inputVocabMatrix, y, discrete_features=True)
               ))

d = Counter(res)
#print(res)
topIndexes = []
for k, v in d.most_common(500):
    topIndexes.append(k)

print(topIndexes)

output = []
rowLen = len(input[0])

res_file = open("Sequence_model_data/train_top.csv", "w")

row_idx = 0
for row in input:
    newRow = []
    for data in row:
        if data in topIndexes:
            newRow.append(data)

    newRowLen = len(newRow)
    diff = rowLen - newRowLen

    if(diff > 0):
        newRow = [0 for n in range(diff)] + newRow
        #print(newRow)

    output.append(newRow)

    res_file.write(str(y[row_idx]))
    for i in range(0, len(newRow)):
        res_file.write(',' + str(newRow[i]))

    res_file.write("\n")
    row_idx = row_idx + 1

res_file.close()











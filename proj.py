# import regular expressins packge
# import numbers package
import numpy as np
import pandas as pd
import re
import math
import xlrd
import csv
import openpyxl

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
import matplotlib.pyplot as plt

def readFile(fileName):
    file = open(fileName,'r',encoding="cp437")
    fileStr = ""
    for line in file:
        fileStr += line
    return fileStr
        
# Remove extra spaces
# Remove non-letter chars    
# Change to lower 
def preProcess(fileStr):
    fileStr = re.sub(" +"," ", fileStr)
    fileStr = re.sub("[^a-zA-Z ]","", fileStr)
    fileStr = fileStr.lower()
    return fileStr

def diff(first, second):
        return [item for item in first if item not in second]

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u"^" + emot + "$", "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

def pad_features(reviews_ints, seq_length):
    
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features


# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())      
loc = (r'C:/Users/Tom/Documents/GitHub/dataMining/dataSet.csv')

# Assign colum names to the dataset
colnames = ['itemID', 'Sentiment', 'SentimentSource', 'SentimentText']

# Read dataset to pandas dataframe
tweetsData = pd.read_csv(loc, names=colnames)

#tweets = sorted(set(tweetsData['SentimentText']), key=tweetsData.index)
################
#REMOVE [0:499]!!!!!!!!!!!!!!!!
tweets = list(tweetsData['SentimentText'][0:499])
del tweets[0]

#splited = [set() for _ in range(len(tweets))]

cleanTweets = [list() for _ in range(len(tweets))]

i=0
for tweet in tweets:
   cleanTweets[i] = list(tweet.split())
   cleanTweets[i] = diff(cleanTweets[i], stopWordsSet)
   if len(cleanTweets[i]) == 0:
       cleanTweets.remove(cleanTweets[i])
   else:
       i += 1

changed = ""
for tweet in cleanTweets:
    i = 0
    for wordInside in tweet:
        changed = convert_emoticons(wordInside)
        tweet[i] = changed
        i += 1



















## Build a dictionary that maps words to integers
################
#REMOVE [0:499]!!!!!!!!!!!!!!!!
tweetsSentiments = list(tweetsData['Sentiment'][0:499])
Counter(tweetsSentiments)
#################


#create a new list for the words
words = []
for sentence in cleanTweets:
    for singleWord in sentence:
        words.append(singleWord)

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {words: ii for ii, words in enumerate(vocab, 1)}

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
tweets_ints = []
for tweet in cleanTweets:
    tweets_ints.append([vocab_to_int[word] for word in tweet])

tweets_padded = pad_features(tweets_ints, seq_length=300)

#Split to training and testing sets

split_frac = 0.7

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(tweets_padded)*split_frac)
train_x, remaining_x = tweets_padded[:split_idx], tweets_padded[split_idx:]
train_y, remaining_y = tweetsSentiments[:split_idx], tweetsSentiments[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

train_x_np = np.array(train_x)
train_y_np = np.empty(len(train_y))
val_x_np = np.array(val_x)
val_y_np = np.empty(len(val_y))
test_x_np = np.array(test_x)
test_y_np = np.empty(len(test_y))

i = 0
for string in train_y[1:]:
    train_y_np[i] = int(string)
    i += 1
    
i = 0
for string in test_y:
    test_y_np[i] = int(string)
    i += 1 
i = 0
for string in val_y:
    val_y_np[i] = int(string)
    i += 1
    
#Reshape the data into 3-D array
train_x_np = np.reshape(train_x_np, (train_x_np.shape[0],train_x_np.shape[1],1))
train_y_np = np.reshape(train_y_np, (train_y_np.shape[0],1))
#val_x_np = np.reshape(val_x_np, (val_x_np.shape[0],val_x_np.shape[1],1))
#test_x_np = np.reshape(test_x_np, (test_x_np.shape[0],test_x_np.shape[1],1))
print(train_x_np[28])
model = Sequential()
model.add(LSTM(8,input_shape=(300,1),return_sequences=False))#True = many to many
model.add(Dense(2,kernel_initializer='normal',activation='linear'))
model.add(Dense(1,kernel_initializer='normal',activation='linear'))
model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
model.fit(train_x_np,train_y_np,epochs=10,batch_size=5,validation_split=0.5,verbose=0);
scores = model.evaluate(train_x_np,train_y_np,verbose=1,batch_size=5)
print('Accurracy: {}'.format(scores[1])) 










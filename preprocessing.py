# import regular expressins packge
# import numbers package
import numpy as np
import pandas as pd
import re
import math
import xlrd
import csv
import openpyxl
from keras.preprocessing.sequence import pad_sequences
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim.models
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
    fileStr = re.sub("[^a-zA-Z:)( ]","", fileStr)
    fileStr = fileStr.lower()
    return fileStr
# Remove extra spaces
# Remove non-letter chars    
# Change to lower 
def preProcess1(fileStr):
    fileStr = re.sub(" +"," ", fileStr)
    fileStr = re.sub("[^a-zA-Z] ","", fileStr)
    fileStr = fileStr.lower()
    return fileStr
def diff(first, second):
        return [item for item in first if item not in second]

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u"^" + emot + "$", " ".join(EMOTICONS[emot].replace(",","").split()), text)
        preProcess1(text)
        arr=text.split(" ")
    return arr[0]

def pad_features(reviews_ints, seq_length):
    
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

def full_pre_process():
    # Read stop words file - words that can be removed
    stopWordsSet = set(readFile('stopwords_en.txt').split())      
    loc = (r'C:\Users\Tom\Downloads\Sentiment-Analysis-Dataset\Sentiment Analysis Dataset.csv')
    
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
       tweet=preProcess(tweet)
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
            changed=preProcess1(changed)
            #changed = list(changed.split())
            #changed = diff(changed, stopWordsSet)
            tweet[i] = changed
            if len(changed)==0:
                tweet.remove(tweet[i])
            else:
                i += 1
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(cleanTweets)
    sequences = tokenizer.texts_to_sequences(cleanTweets)
    tweets = pad_sequences(sequences, maxlen=200)
    
    
    
    return cleanTweets,tweets,tweetsData['Sentiment'][1:490]

  





















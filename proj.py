# import regular expressins packge
# import numbers package
import numpy as np
import pandas as pd
import re
import xlrd
import csv
import openpyxl
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

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

def diff(first, second):l
        return [item for item in first if item not in second]

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u"^" + emot + "$", "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text


# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())      
loc = (r'C:/Users/Tzilb/Documents/dataMaining/Project/Sentiment Analysis Dataset.csv')

# Assign colum names to the dataset
colnames = ['itemID', 'Sentiment', 'SentimentSource', 'SentimentText']

# Read dataset to pandas dataframe
tweetsData = pd.read_csv(loc, names=colnames)

#tweets = sorted(set(tweetsData['SentimentText']), key=tweetsData.index)
tweets = list(tweetsData['SentimentText'])
del tweets[0]

#splited = [set() for _ in range(len(tweets))]

cleanTweets = [list() for _ in range(len(tweets))]

i=0
for tweet in tweets:
   cleanTweets[i] = list(tweet.split())
   cleanTweets[i] = diff(cleanTweets[i], stopWordsSet)
   i += 1

changed = ""
for tweet in cleanTweets:
    i = 0
    for wordInside in tweet:
        changed = convert_emoticons(wordInside)
        tweet[i] = changed
        i += 1


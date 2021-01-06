# import regular expressins packge
# import numbers package
import numpy as np
import pandas as pd
import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

def full_pre_process():
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
    max_features=2000
    tokenizer=Tokenizer(num_words=max_features,split='')
    tokenizer.fit_on_texts(cleanTweets)
    X=tokenizer.texts_to_sequences(cleanTweets)
    X=pad_sequences(X)
    return X,tweetsData['Sentiment'][1:]
X,sentiment=full_pre_process()




























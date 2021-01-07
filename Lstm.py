# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:38:23 2021

@author: Tom
"""

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
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation,Embedding,SpatialDropout1D
from preprocessing import full_pre_process
import matplotlib.pyplot as plt

def LstmModel():
    max_features=2000
    X,sentiment=full_pre_process()
    embed_dim=128
    lstm_out=196
    model=Sequential()
    model.add(Embedding(max_features,embed_dim,input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out))
    
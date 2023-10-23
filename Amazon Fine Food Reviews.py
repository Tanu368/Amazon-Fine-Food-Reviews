import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import pandas as pd
data=pd.read_csv("/gdrive/MyDrive/Reviews.csv")
data

data.shape
def partition(x):
  if x < 3 :
    return 0
  return 1


data['Sentiment'] = data['Score'].apply(partition)
print(data['Sentiment'].shape)
print(data.head(3))

data[data.duplicated(subset = [ 'ProductId', 'ProfileName', 'Score', 'Time', 'Summary'], keep = False)]
data[data.duplicated(subset = [ 'ProductId', 'ProfileName', 'Score', 'Time', 'Summary'], keep = False)].loc[data.ProductId == 'B005K4Q1VI', :]

#Data Cleaning- Removing Duplicates
#Sorting data according to ProductId in ascending order
sorted_data=data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape

#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(data['Id'].size*1.0)*100

#Checking some constraints like helpfullness numerator should be greater than denominator. So checking on them

final[final['HelpfulnessNumerator'] > final['HelpfulnessDenominator']]
final = final[final['HelpfulnessNumerator'] <= final['HelpfulnessDenominator']]
final.reset_index(inplace=True)
final.shape

#Preprocessing

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


# https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
from bs4 import BeautifulSoup

# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase



    # https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
# <br /><br /> ==> after the above steps, we are getting "br br"
# we are including them into stop words list
# instead of <br /> if we have <br/> these tags would have revmoved in the 1st step

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])


            # Combining all the above steps
from tqdm import tqdm
preprocessed_reviews = []
# tqdm is for printing the status bar
for sentance in tqdm(final['Text'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews.append(sentance.strip())

    preprocessed_reviews[1]

    preprocessed_reviews[15000]

    len(preprocessed_reviews)

    #Featurization
    sample_data = final.sample(n = 5000)
sample_data.head()
sample_data.drop(columns = ['Id', 'ProductId', 'UserId','ProfileName', 'Time', 'Summary', 'Score'], inplace=True)
sample_data.head(3)

sample_data.index.values

sample_reviews = [ preprocessed_reviews[i] for i in sample_data.index.values]
sample_reviews[0]

sample_data['preprocessed'] = sample_reviews
sample_data.head()

sample_data.drop(columns = ['Text', 'index'], inplace=True)

sample_data.head(3)

from sklearn.model_selection import train_test_split

y = sample_data['Sentiment'].values
X = sample_data.drop(columns =['Sentiment'])

print(X.shape)
print(y.shape)

sample_data['Sentiment'].value_counts()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 24)

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
count_vect.fit(train_X['preprocessed'])

print("some feature names ", count_vect.get_feature_names()[:10])
print('='*50)

final_vectors = count_vect.transform(train_X['preprocessed'])

final_vectors.shape

num_feats = train_X[['HelpfulnessNumerator' ,	'HelpfulnessDenominator']].values

from scipy import sparse

training_data = sparse.hstack(( num_feats, final_vectors))

training_data.shape

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 15000)

model.fit(training_data, train_y)

# predict on test datasets

final_test_vectors = count_vect.transform(test_X['preprocessed'].values)
final_test_vectors.shape

test_feats = test_X[['HelpfulnessNumerator' ,	'HelpfulnessDenominator']].values

test_data = sparse.hstack(( test_feats, final_test_vectors))

test_data.shape

from sklearn.metrics import accuracy_score

preds = model.predict(test_data)

# get the actual values
y_true = test_y

accuracy_score(y_true, preds)
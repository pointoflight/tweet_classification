#!/usr/bin/env python
# coding: utf-8

# In[123]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import nltk
nltk.download('words')
nltk.download('stopwords')
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import numpy as np
import re
import preprocessor as p
get_ipython().run_line_magic('matplotlib', 'inline')


# In[124]:


tweets = pd.read_csv('socialmedia-disaster-tweets-DFE.csv', encoding = 'latin-1')
tweets.head()


# In[125]:


tweets.rename(columns = {'choose_one': 'labels', 'text': 'message'}, inplace = True)
tweets.head()


# In[126]:


tweets.drop(['_golden', '_unit_id', '_unit_state','_trusted_judgments','_last_judgment_at','choose_one:confidence','choose_one_gold','keyword','location','tweetid','userid'], axis = 1, inplace = True)
tweets.head()


# In[127]:


tweets['label'] = tweets['labels'].map({'Relevant': 1, 'Not Relevant': 0})
tweets.drop(['labels'], axis = 1, inplace = True)
tweets


# In[128]:


words = set(nltk.corpus.words.words())
for index, row in tweets.iterrows():
    tweets.set_value(index, 'message', p.clean(row['message']))
    tweets.set_value(index, 'message', BeautifulSoup(row['message'], 'lxml'))
    tweets.set_value(index, 'message', re.sub(r'@[A-Za-z0-9]+','',row['message']))
    tweets.set_value(index, 'message', re.sub('https?://[A-Za-z0-9./]+','',row['message']))
    tweets.set_value(index, 'message', re.sub("[^a-zA-Z]", " ", row['message']))

tweets


# In[129]:



trainIndex, testIndex = list(), list()
for i in range(tweets.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = tweets.loc[trainIndex]
testData = tweets.loc[testIndex]
trainData.size


# In[130]:


trainData.reset_index(inplace = True)
trainData.drop(['index'], axis = 1, inplace = True)
trainData.head()


# In[131]:


testData.reset_index(inplace = True)
testData.drop(['index'], axis = 1, inplace = True)
testData.head()


# In[132]:


disaster_words = ' '.join(list(tweets[tweets['label'] == 1]['message']))
disaster_wc = WordCloud(width = 512,height = 512).generate(disaster_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(disaster_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[133]:


normal_words = ' '.join(list(tweets[tweets['label'] == 0]['message']))
normal_wc = WordCloud(width = 512,height = 512).generate(normal_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(normal_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[134]:


def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words


# In[135]:


class DisasterClassifier(object):
    def __init__(self, trainData, method = 'tf-idf'):
        self.tweets, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_disaster = dict()
        self.prob_normal = dict()
        for word in self.tf_disaster:
            self.prob_disaster[word] = (self.tf_disaster[word] + 1) / (self.disaster_words +                                                                 len(list(self.tf_disaster.keys())))
        for word in self.tf_normal:
            self.prob_normal[word] = (self.tf_normal[word] + 1) / (self.normal_words +                                                                 len(list(self.tf_normal.keys())))
        self.prob_disaster_mail, self.prob_normal_mail = self.disaster_tweets / self.total_tweets, self.normal_tweets / self.total_tweets 


    def calc_TF_and_IDF(self):
        noOfMessages = self.tweets.shape[0]
        self.disaster_tweets, self.normal_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_tweets = self.disaster_tweets + self.normal_tweets
        self.disaster_words = 0
        self.normal_words = 0
        self.tf_disaster = dict()
        self.tf_normal = dict()
        self.idf_disaster = dict()
        self.idf_normal = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.tweets[i])
            count = list() #To keep track of whether the word has ocured in the message or not.
                           #For IDF
            for word in message_processed:
                if self.labels[i]:
                    self.tf_disaster[word] = self.tf_disaster.get(word, 0) + 1
                    self.disaster_words += 1
                else:
                    self.tf_normal[word] = self.tf_normal.get(word, 0) + 1
                    self.normal_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_disaster[word] = self.idf_disaster.get(word, 0) + 1
                else:
                    self.idf_normal[word] = self.idf_normal.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_disaster = dict()
        self.prob_normal = dict()
        self.sum_tf_idf_disaster = 0
        self.sum_tf_idf_normal = 0
        for word in self.tf_disaster:
            self.prob_disaster[word] = (self.tf_disaster[word]) * log((self.disaster_tweets + self.normal_tweets)                                                           / (self.idf_disaster[word] + self.idf_normal.get(word, 0)))
            self.sum_tf_idf_disaster += self.prob_disaster[word]
        for word in self.tf_disaster:
            self.prob_disaster[word] = (self.prob_disaster[word] + 1) / (self.sum_tf_idf_disaster + len(list(self.prob_disaster.keys())))
            
        for word in self.tf_normal:
            self.prob_normal[word] = (self.tf_normal[word]) * log((self.disaster_tweets + self.normal_tweets)                                                           / (self.idf_disaster.get(word, 0) + self.idf_normal[word]))
            self.sum_tf_idf_normal += self.prob_normal[word]
        for word in self.tf_normal:
            self.prob_normal[word] = (self.prob_normal[word] + 1) / (self.sum_tf_idf_normal + len(list(self.prob_normal.keys())))
            
    
        self.prob_disaster_mail, self.prob_normal_mail = self.disaster_tweets / self.total_tweets, self.normal_tweets / self.total_tweets 
                    
    def classify(self, processed_message):
        pdisaster, pnormal = 0, 0
        for word in processed_message:                
            if word in self.prob_disaster:
                pdisaster += log(self.prob_disaster[word])
            else:
                if self.method == 'tf-idf':
                    pdisaster -= log(self.sum_tf_idf_disaster + len(list(self.prob_disaster.keys())))
                else:
                    pdisaster -= log(self.disaster_words + len(list(self.prob_disaster.keys())))
            if word in self.prob_normal:
                pnormal += log(self.prob_normal[word])
            else:
                if self.method == 'tf-idf':
                    pnormal -= log(self.sum_tf_idf_normal + len(list(self.prob_normal.keys()))) 
                else:
                    pnormal -= log(self.normal_words + len(list(self.prob_normal.keys())))
            pdisaster += log(self.prob_disaster_mail)
            pnormal += log(self.prob_normal_mail)
        return pdisaster >= pnormal
    
    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


# In[136]:


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


# In[137]:


dc_tf_idf = DisasterClassifier(trainData, 'tf-idf')
dc_tf_idf.train()
preds_tf_idf = dc_tf_idf.predict(testData['message'])
metrics(testData['label'], preds_tf_idf)


# In[138]:


dc_bow = DisasterClassifier(trainData, 'bow')
dc_bow.train()
preds_bow = dc_bow.predict(testData['message'])
metrics(testData['label'], preds_bow)


# In[ ]:





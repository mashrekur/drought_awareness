#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl
import matplotlib.pyplot as plt
import pingouin as pg


# In[2]:


#Load dataframe 
with open('data/drought_twitter_corpus_df.pkl', 'rb') as f:
    corpus_df = pkl.load(f)


# In[3]:


# sentiment_list = []

# def percentage(part,whole):
#  return 100 * float(part)/float(whole)

# # noOfTweet = int(input ('Please enter how many tweets to analyze: '))

# positive = 0
# negative = 0
# neutral = 0
# polarity = 0
# tweet_list = []
# neutral_list = []
# negative_list = []
# positive_list = []

# for twt in corpus_df['tweet']:
# #     print(twt)
    
#     tweet_list.append(twt)
#     analysis = TextBlob(twt)
#     score = SentimentIntensityAnalyzer().polarity_scores(twt)
#     neg = score['neg']
#     neu = score['neu']
#     pos = score['pos']
#     comp = score['compound']
#     polarity += analysis.sentiment.polarity
 
#     if neg > pos:
#         sentiment_list.append('neg')
# #         negative_list.append(twt)
# #         negative += 1
#     elif pos > neg:
#         sentiment_list.append('pos')
# #         positive_list.append(twt)
# #         positive += 1
#     elif pos == neg:
#         sentiment_list.append('neu')
# #         neutral_list.append(twt)
# #         neutral += 1
# # positive = percentage(positive, noOfTweet)
# # negative = percentage(negative, noOfTweet)
# # neutral = percentage(neutral, noOfTweet)
# # polarity = percentage(polarity, noOfTweet)
# # positive = format(positive, '.1f')
# # negative = format(negative, '.1f')
# # neutral = format(neutral, '.1f')


# In[4]:


# senti_arr = np.array(sentiment_list)
# np.save('data/senti_array', senti_arr, allow_pickle=True, fix_imports=True)


# In[5]:


# #Number of Tweets (Total, Positive, Negative, Neutral)
# tweet_list = pd.DataFrame(tweet_list)
# neutral_list = pd.DataFrame(neutral_list)
# negative_list = pd.DataFrame(negative_list)
# positive_list = pd.DataFrame(positive_list)
# print('total number: ',len(tweet_list))
# print('positive number: ',len(positive_list))
# print('negative number: ', len(negative_list))
# print('neutral number: ',len(neutral_list))


# In[6]:


# #Creating PieCart
# labels = ['Positive [‘+str(positive)+’%]' , 'Neutral [‘+str(neutral)+’%]','Negative [‘+str(negative)+’%]']
# sizes = [positive, neutral, negative]
# colors = ['yellowgreen', 'blue','red']
# patches, texts = plt.pie(sizes,colors=colors, startangle=90)
# plt.style.use('default')
# plt.legend(labels)
# plt.title('Sentiment Analysis Result for keyword= “+Drought+”')
# plt.axis('equal')
# plt.show()


# In[7]:


senti_arr = np.load('data/senti_array.npy')


# In[8]:


senti_list = senti_arr.tolist()


# In[9]:


corpus_df['sentiment'] =  senti_list


# In[10]:


month_list = []

for n, twt in enumerate(corpus_df['id']):
    month_list.append(corpus_df['created_at'][n][:7])

    
corpus_df['Month'] = month_list


# In[11]:


#Analyse the variability of sentiments

#takes all unique values in data - year as well as how often they occur and returns them as an array.
uniqueyears, time_slices = np.unique(corpus_df['Month'], return_counts=True) 

time_slices = time_slices.tolist()

def cumSum(s):
    sm=0
    cum_list=[]
    for i in s:
        sm=sm+i
        cum_list.append(sm)
    return cum_list

time_slices_cum = cumSum(time_slices)
time_slices_cum


# In[12]:


#Add 0 as the first element of the cumulative document array
a=0
time_slices_cum.insert(0,a)
time_slices_cum


# In[13]:


def percentage(part,whole):
    return 100 * float(part)/float(whole)


pos_list = []
neg_list = []
neu_list = []




for t in range(len(time_slices)):
    
    positive = 0
    negative = 0
    neutral = 0
    
    for i in range(time_slices_cum[t], time_slices_cum[t+1]):
        if str(corpus_df['sentiment'][i]) == 'pos':
            positive += 1
        elif str(corpus_df['sentiment'][i]) == 'neg':
            negative += 1
        elif str(corpus_df['sentiment'][i]) == 'neu':
            neutral += 1

    pos = percentage(positive, time_slices[t])
    neg = percentage(negative, time_slices[t])
    neu = percentage(neutral, time_slices[t])
    
    pos_list.append(pos)
    neg_list.append(neg)
    neu_list.append(neu)


# In[14]:


#Pull months
months = np.unique(corpus_df['Month'])
months


# In[30]:


#Plot sentiment variability

n = 10  # Show every nth label
plt.figure(figsize=(50,15))
plt.plot(months, pos_list, color = 'green', label = 'Positive', linewidth = 5)
plt.plot(months, neg_list, color = 'red', label = 'Negative', linewidth = 5)
plt.plot(months, neu_list, color = 'orange', label = 'Neutral', linewidth = 5)

# Reduce the density of xticks
ticks = plt.gca().get_xticks()
ticklabels = [months[int(i)] if i < len(months) and i%10 == 0 else '' for i in ticks]

plt.gca().set_xticklabels(ticklabels, fontsize=40, rotation=90)

plt.yticks(size = 40)
plt.xlim(0,155)
plt.grid()
plt.xlabel("Months", fontsize = 40)
plt.ylabel("Sentiment Percentages", fontsize = 40)
plt.legend(prop={'size': 40}, fancybox = True, loc = 'upper left')
plt.title("Variability of Public Sentiment of Drought from Twitter Data", size = 45)
plt.savefig('sentiment_variability')
plt.show()


# In[ ]:


# Testing for significant trends

print('Positive Sentiment')
pg.corr(range(len(months)), np.array(pos_list))


# In[ ]:


print('Negative Sentiment')
pg.corr(range(len(months)), np.array(neg_list))


# In[ ]:


print('Neutral Sentiment')
pg.corr(range(len(months)), np.array(neu_list))


# In[ ]:


#Slightly increasing statistically significant trends observed for Positive and Negative Sentiments towards Drought
#Moderately decreasing statistically significant trend observed for Neutral Sentiment towards Drought
#Increasing polarization?
#Correlate with Google Search Interest?


# In[ ]:


#Load Worldwide Google Search Interest for Drought 
google_drought = pd.read_csv('data/google_drought_worldwide.csv')
google_climatechange = pd.read_csv('data/climatechange_google.csv')
google_waterscarcity = pd.read_csv('data/waterscarcity_google.csv')
google_drought = google_drought['Category: All categories'][1:]
google_climatechange = google_climatechange['Category: All categories'][1:]
google_waterscarcity = google_waterscarcity['Category: All categories'][1:]


# In[ ]:


google_drought_list = []
google_climatechange_list = []
google_waterscarcity_list = []

for d in range(len(google_drought)):
    
    google_drought_list.append(int(google_drought[d]))
    google_climatechange_list.append(int(google_climatechange[d]))
    google_waterscarcity_list.append(int(google_waterscarcity[d]))


# In[ ]:


fig = plt.figure()
fig, ax1 = plt.subplots(figsize=(50, 15))
ax2 = ax1.twinx()

ax1.tick_params(axis='both', which='major', labelsize=25, rotation = 90)
ax2.tick_params(axis='both', which='major', labelsize=25)

ax1.plot(months, pos_list, color = 'green', label = 'Positive', linewidth = 5)
ax1.plot(months, neg_list, color = 'red', label = 'Negative', linewidth = 5)
ax1.plot(months, neu_list, color = 'orange', label = 'Neutral', linewidth = 5)
ax2.plot(months, google_drought_list, color = 'skyblue', linestyle='dashed', linewidth = 7, label = 'Google Drought SI')
ax2.plot(months, google_climatechange_list, color = 'gray', linestyle='dashed', linewidth = 7, label = 'Google Climate Change SI')
ax2.plot(months, google_waterscarcity_list, color = 'blue', linestyle='dashed', linewidth = 7, label = 'Google Water Scarcity SI')

plt.xlim(0,168)
ax1.set_ylabel('Sentiment Percentage', size = 40)
ax2.set_ylabel('Google SI', size = 40)
ax1.set_xlabel('Months', size = 40)
ax1.grid()
fig.legend(loc = 'upper right', fancybox = True, shadow = True, prop={'size': 20})
plt.title('Variability of Global Twitter Public Sentiment and Google Public Search Interest on Drought', fontsize=50)


# In[ ]:


#Test for statistical relationship between Twitter Sentiment and Google Search Interest

print('Positive Sentiment & Google Drought SI')
pg.corr(np.array(pos_list), np.array(google_drought_list))


# In[ ]:


print('Negative Sentiment & Google Drought SI')
pg.corr(np.array(neg_list), np.array(google_drought_list))


# In[ ]:


print('Neutral Sentiment & Google Drought SI')
pg.corr(np.array(neu_list), np.array(google_drought_list))


# In[ ]:


#Statistically significant weak positive correlation between google drought SI and polar sentiments
#Statistically significant moderate negative correlation between google SI and neutrality
#It appears polarization is statistically correlated to Google SI


# In[ ]:


print('Positive Sentiment & Google Climate Change SI')
pg.corr(np.array(pos_list), np.array(google_climatechange_list))


# In[ ]:


print('Negative Sentiment & Google Climate Change SI')
pg.corr(np.array(neg_list), np.array(google_climatechange_list))


# In[ ]:


print('Neutral Sentiment & Google Climate Change SI')
pg.corr(np.array(neu_list), np.array(google_climatechange_list))


# In[ ]:


print('Positive Sentiment & Google Water Scarcity SI')
pg.corr(np.array(pos_list), np.array(google_waterscarcity_list))


# In[ ]:


print('Negative Sentiment & Google Water Scarcity SI')
pg.corr(np.array(neg_list), np.array(google_waterscarcity_list))


# In[ ]:


print('Neutral Sentiment & Google Water Scarcity SI')
pg.corr(np.array(neu_list), np.array(google_waterscarcity_list))


# In[ ]:


corpus_df


# In[ ]:


corpus_df[9:10]


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from IPython.display import Image
import requests
import json
import csv
import dateutil
import pandas as pd
import time
from calendar import monthrange
import numpy as np


# In[2]:


os.environ['TOKEN'] = ''


# In[3]:


# create our auth() function, which retrieves the token from the environment.

def auth():
    return os.getenv('TOKEN')


# In[4]:


#take bearer token, pass it for authorization and return headers we will use to access the API

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


# In[5]:


# build the request for the endpoint we are going to use and the parameters we want to pass

def create_url(keyword, start_date, end_date, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/all" #Change to the endpoint we want to collect data from

    #change params based on the endpoint we are using
    query_params = {'query': keyword,
                    'start_time': start_date, #YYYY-MM-DDTHH:mm:ssZ 
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

Image(filename='data/api_params.png') 


# In[6]:


# send the “GET” request and if everything is correct (response code 200), it will return the response in “JSON” format

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


# In[7]:


#Inputs for the request
# bearer_token = auth()
# headers = create_headers(bearer_token)
# keyword = "drought lang:en"
# start_time = "2022-08-01T00:00:00.000Z"
# end_time = "2022-08-31T00:00:00.000Z"
# max_results = 500


# In[8]:


# url = create_url(keyword, start_time,end_time, max_results)
# json_response = connect_to_endpoint(url[0], headers, url[1])


# In[9]:


# print(json.dumps(json_response, indent=4, sort_keys=True))


# In[10]:


# json_response['data'][0]['created_at']


# In[11]:


# json_response['meta']['result_count']


# In[12]:


# with open('data/data.json', 'w') as f:
#     json.dump(json_response, f)


# In[13]:


#Save Results to CSV for analysis
# Create file
# csvFile = open("data/data.csv", "a", newline="", encoding='utf-8')
# csvWriter = csv.writer(csvFile)

# #Create headers for the data you want to save, in this example, we only want save these columns in our dataset
# csvWriter.writerow(['author id', 'created_at', 'geo', 'id','lang', 'like_count', 'quote_count', 'reply_count','retweet_count','source','tweet'])
# csvFile.close()


# In[14]:



def append_to_csv(json_response, fileName):

    #A counter variable
    counter = 0

    #Open OR create the target CSV file
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #Loop through each tweet
    for tweet in json_response['data']:
        
        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 1. Author ID
        author_id = tweet['author_id']

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])

        # 3. Geolocation
        if ('geo' in tweet):
            try:
                geo = tweet['geo']['place_id']
            except:
                geo = tweet['geo']            
        else:
            geo = " "

        # 4. Tweet ID
        tweet_id = tweet['id']

        # 5. Language
        lang = tweet['lang']

        # 6. Tweet metrics
        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']

        # 7. source
        try:
            source = tweet['source']
        except:
            source = 'NaN'

        # 8. Tweet text
        text = tweet['text']
        
        # Assemble all data in a list
        res = [author_id, created_at, geo, tweet_id, lang, like_count, quote_count, reply_count, retweet_count, source, text]
        
        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter)


# In[15]:


# append_to_csv(json_response, "data/data.csv")


# In[16]:


# #Load dataframe 
# proto_tweets = pd.read_csv('data/data.csv')
# proto_tweets


# In[17]:


#get start and end list
start_list = []
end_list = []



for y in range(2008,2022,1):
    for m in range(1,13,1):
        for d in range(1,(monthrange(y,m)[1]+1)):
            if len(str(m)) == 1 and len(str(d)) == 1:
                start_list.append(str(y)+'-0'+str(m)+'-0'+str(d)+'T00:00:00.000Z')
            elif len(str(m)) == 1 and len(str(d)) == 2:
                start_list.append(str(y)+'-0'+str(m)+'-'+str(d)+'T00:00:00.000Z')
            elif len(str(m)) == 2 and len(str(d)) == 2:
                start_list.append(str(y)+'-'+str(m)+'-'+str(d)+'T00:00:00.000Z')
                
for y in range(2008,2022,1):
    for m in range(1,3,1):
        for d in range(1,(monthrange(y,m)[1]+1)):
            if len(str(m)) == 1 and len(str(d)) == 1:
                end_list.append(str(y)+'-0'+str(m)+'-0'+str(d)+'T23:59:00.000Z')
            elif len(str(m)) == 1 and len(str(d)) == 2:
                end_list.append(str(y)+'-0'+str(m)+'-'+str(d)+'T23:59:00.000Z')
            elif len(str(m)) == 2 and len(str(d)) == 2:
                end_list.append(str(y)+'-'+str(m)+'-'+str(d)+'T23:59:00.000Z')


# In[18]:


start_list


# In[19]:


#Inputs for tweets
bearer_token = auth()
headers = create_headers(bearer_token)
keyword = "climate lang:en"

max_results = 500

#Total number of tweets we collected from the loop
total_tweets = 0

# Create file
csvFile = open("data/climate_twitter_data_2008_2022.csv", "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)

#Create headers for the data you want to save, in this example, we only want save these columns in our dataset
csvWriter.writerow(['author id', 'created_at', 'geo', 'id','lang', 'like_count', 'quote_count', 'reply_count','retweet_count','source','tweet'])
csvFile.close()

for i in range(0,len(start_list)):

    # Inputs
    count = 0 # Counting tweets per time period
    max_count = 100 # Max tweets per time period
    flag = True
    next_token = None
    
    # Check if flag is true
    while flag:
        # Check if max_count reached
        if count >= max_count:
            break
        print("-------------------")
        print("Token: ", next_token)
        url = create_url(keyword, start_list[i],end_list[i], max_results)
        json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
        result_count = json_response['meta']['result_count']

        if 'next_token' in json_response['meta']:
            # Save the token to use for next call
            next_token = json_response['meta']['next_token']
            print("Next Token: ", next_token)
            if result_count is not None and result_count > 0 and next_token is not None:
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "data/climate_twitter_data_2008_2022.csv")
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(3)                
        # If no next token exists
        else:
            if result_count is not None and result_count > 0:
                print("-------------------")
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "data/climate_twitter_data_2008_2022.csv")
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(3)
            
            #Since this is the final request, turn flag to false to move to the next time period.
            flag = False
            next_token = None
        time.sleep(3)
print("Total number of results: ", total_tweets)


# In[ ]:


#Load dataframe 
proto_tweets_2022 = pd.read_csv('data/climate_twitter_data_2008_2022.csv')
proto_tweets_2022


# In[ ]:


month_list = []

for n, twt in enumerate(proto_tweets_2022['id']):
    month_list.append(proto_tweets_2022['created_at'][n][:7])


# In[ ]:


proto_tweets_2022['Month'] = month_list


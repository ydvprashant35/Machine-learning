#!/usr/bin/env python
# coding: utf-8

# In[204]:


import pandas as pd
import numpy as np


# In[205]:


movies=pd.read_csv("C:/Users/Admin/Desktop/tmdb_5000_movies.csv/tmdb_5000_movies.csv",encoding='ISO-8859-1')
credits=pd.read_csv("C:\\Users\\Admin\\Desktop\\tmdb_5000_credits.csv")


# In[206]:


movies.head()


# In[207]:


credits.head()


# In[208]:


credits.shape


# In[209]:


# we are makeing a movie recomendation system based on contant.
# Now we need to merging both the datasets with respect to the title column.
movies=movies.merge(credits,on='title')
movies.shape


# In[210]:


# Now,we are going to select the columns that are usefull and remove other columns.
# we are selecting the columns that are helpfull to make tages.
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[211]:


# Now we are going to make a new dataframe which hane columns movies_id,title,tags.
# For tags we are going to merge the overview,genres,keywords,cast,crew columns.


# In[212]:


# firstly we are going to check that there is not any duplicate value or null values in the dataset.
movies.isnull().sum()


# In[213]:


movies.dropna(inplace=True)


# In[214]:


movies.duplicated().sum()


# In[215]:


movies.iloc[0].genres


# In[216]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# we want in this way-['Action','Adventeger','Fantasy','Science Fiction']


# In[217]:


import ast #it is use to conver the string into numbers
def convert (obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[218]:


movies['genres']=movies['genres'].apply(convert)


# In[219]:


movies.head()


# In[220]:


movies['keywords']=movies['keywords'].apply(convert)
movies.head()


# In[221]:


import ast
def convert3 (obj):
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!=3:
            L.append(i['name'])
            count+=1
        else:
            break
    return L    


# In[222]:


movies['cast']=movies['cast'].apply(convert3)
movies['cast']


# In[223]:


movies


# In[224]:


import ast #it is use to conver the string into numbers
def fetch_director (obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L   


# In[225]:


movies['crew']=movies['crew'].apply(fetch_director)
movies['crew']


# In[226]:


movies.head()


# In[227]:


movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['overview']


# In[228]:


movies.head()


# In[229]:


# Now we need to apply some transformation on genres,keywords,cast,crew.In transformation we are going to remove the space 
# between words like Sam Worthington to SamWorthington.
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[230]:


movies.head()


# In[231]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[232]:


movies.head()


# In[233]:


new_df=movies[['movie_id','title','tags']]
new_df.head()


# In[234]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']


# In[235]:


new_df.head()


# In[186]:


#!pip install nltk


# In[242]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[243]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[244]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[245]:


new_df['tags']=new_df['tags'].apply(stem)


# In[246]:


new_df['tags'][0]


# In[247]:


# Now we need to find out the relation between different-different tags.so, for this we first convert the tags in to vectors.
# we have 4784 tags after converting them into vectors we check the closest vectors.
# so,for this we use the technique know as Bag of words.


# In[248]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
vectors.shape


# In[249]:


cv.get_feature_names_out()


# In[250]:


# now we check the distance of one movie vector to another.
# And we calculate the cossin distance of vectors.


# In[251]:


from sklearn.metrics.pairwise import cosine_similarity


# In[256]:


similarity=cosine_similarity(vectors)


# In[262]:


similarity[0]


# In[261]:


#this is the distance of first movie vector with another movie vector
list(enumerate(similarity[0]))


# In[265]:


#this function returns the five similar movies with respect to the movie enteredd in the function. 
def recommend(movie):
    movie_index=new_df[new_df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[266]:


recommend('Avatar')


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:12:01 2018

@author: Administrator
"""
# _*_ coding:utf-8 _*_
import os
import sklearn.datasets
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
english_stemmer=nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
      def build_analyzer(self):
          analyzer=super(TfidfVectorizer,self).build_analyzer()
          return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))    
DIR=r'E:\data'
i=0
df_empty = pd.DataFrame(columns=['Newsgroup', 'Document_id', 'News']) 
split=re.compile(r'Newsgroup:|document_id:')
for f in os.listdir(DIR):
    if 'txt' in f:
        try:
            with open(os.path.join(DIR,f),'r', encoding='UTF-8') as f1:
                text=''
                tmplist=[]
                for line in f1.readlines():
                   line=line.strip()
                   if not line:
                       continue 
                   if split.match(line):
                       if len(tmplist)==2:
                           row=pd.DataFrame([tmplist+[text]],columns=['Newsgroup', 'Document_id', 'News']) 
                           df_empty=df_empty.append(row,ignore_index=True)
                           tmplist=[]
                           text=''
                       if len(tmplist)==0 and re.match('Newsgroup:',line):
                           tmplist=tmplist+[re.sub(r'Newsgroup:', "", line)]
                       if len(tmplist)==1 and re.match(r'document_id:',line):
                           tmplist=tmplist+[re.sub(r'document_id:', "", line)]
                   elif tmplist:
                       text=text+line
        except Exception as e:
                with open(os.path.join(DIR,f),'r', encoding='ISO-8859-1') as f1:
                    text=''
                    tmplist=[]
                    for line in f1.readlines():
                       line=line.strip()
                       if not line:
                           continue 
                       if split.match(line):
                           if len(tmplist)==2:
                               row=pd.DataFrame([tmplist+[text]],columns=['Newsgroup', 'Document_id', 'News']) 
                               df_empty=df_empty.append(row,ignore_index=True)
                               tmplist=[]
                               text=''
                           if len(tmplist)==0 and re.match('Newsgroup:',line):
                               tmplist=tmplist+[re.sub(r'Newsgroup:', "", line)]
                           if len(tmplist)==1 and re.match(r'document_id:',line):
                               tmplist=tmplist+[re.sub(r'document_id:', "", line)]
                       elif tmplist:
                           text=text+line
df_empty['Document_id']=df_empty['Document_id'].astype('int')
vectorizer=StemmedTfidfVectorizer(min_df=10,max_df=0.5,stop_words='english',decode_error='ignore')
vectorized=vectorizer.fit_transform(df_empty['News'])
num_samples,num_features=vectorized.shape
print("sample:%d,feature:%d"%(num_samples,num_features))
               


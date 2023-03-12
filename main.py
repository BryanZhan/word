################## 文本分析 #########################
import jieba
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

# 分词
with open(r'C:\Users\lee\Desktop\s.txt','r',encoding='utf-8') as f:
    txt = f.read()

b = jieba.lcut(txt)

# 去停词，求差集
stopwords=[]
with open(r'C:\Users\lee\Desktop\stopwords.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        l=line.strip()
        if l =='\\n':
            l='\n'
        if l =='\\u3000':
            l='\u3000'
        stopwords.append(l)

x= np.array(b)
y= np.array(stopwords)
z= x[~np.in1d(x,y)]

# 去掉一个字的词
k = [i for i in z if len(i)>1]

# 词频统计分析，但准确度不高
data = pd.DataFrame(k)
data[0].value_counts()[:5]

# TF-IDF，在本文出现频次高，在别人文件出现少
jieba.analyse.extract_tags(txt,topK=20, withWeight=True)

# textrank分析，一个词重不重要主要看周围的词重不重要
jieba.analyse.textrank(txt)

# 查看词性：名词、动词、形容词
list(jieba.posseg.cut(txt))

# 设置语料库
jieba.analyse.set_idf_path

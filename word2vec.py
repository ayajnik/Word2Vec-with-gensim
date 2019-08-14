#importing libraries
import os
import numpy as np
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities
print('\n')
print('Libraries Imported.')
print('\n')

df = str(pd.read_csv('word2vec.csv'));
print('\n')
print('File Uploaded.')
print('\n')
print(df.head(5))

nltk.download('punkt')

#tokenizing
corp_tok = nltk.word_tokenize(df)

#model making
model = gensim.models.Word2Vec([corp_tok], min_count=1, size = 1)

#saving the model
model.save('testmodel')

model = gensim.models.Word2Vec.load('testmodel')

model.train([["speed", "internet"]], total_examples=1, epochs=1)

model.most_similar(['Internet'])

vocabulary = list(model.wv.vocab)
#print(vocabulary)
e = list(model.wv.index2entity[:20])
e

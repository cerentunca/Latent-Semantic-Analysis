
# coding: utf-8

# In[1]:

import io
def get_unigrams(file_name):
    unigrams = {}
    count = 0
    with io.open(file_name, encoding='utf8', errors='ignore') as f:
        for line in f:
            tokens = line.strip().split()
            count+=1
            for token in tokens:
                token = token.lower()
                try:
                    unigrams[token]
                except:
                    unigrams[token] = 0
                unigrams[token] += 1
                
    return unigrams

def index_unigrams(unigrams):
    new_unigrams = {}
    reverse_unigrams = {}
    for index, unigram in enumerate(unigrams):
        new_unigrams[unigram] = index
        reverse_unigrams[index] = unigram
    return new_unigrams, reverse_unigrams
            


# In[28]:

file_name = "NNN/sample_corpus/sample_corpus.txt"
import nltk
import copy
unigrams = get_unigrams(file_name)
words = [i for i in unigrams.keys()]
pos = nltk.pos_tag(words)
verbs = [i[0] for i in pos if i[1]=='VB' or i[1]=='VBD' or i[1]=='VBG' or i[1]=='VBN' or i[1]=='VBP' or i[1]=='VBZ']
iunigrams,runigrams = index_unigrams(unigrams)
unigrams = sorted(unigrams.items(), key = lambda x: x[1], reverse = True )
print (unigrams[0])
#unigrams = [i for i in unigrams if i[0] in verbs]
from pprint import pprint
#pprint.pprint(iunigrams) # Figure out non-stop words
dimensions = [x[0] for x in unigrams[100:1600]]
# count = 0
# dimensions = list()
# for x in unigrams[100:]:
#     if x[0] in verbs:
#         dimensions.append(x[0])
#         count += 1
#     if count == 3000:
#         break
idimensions = {x: index for index, x in enumerate(dimensions)}
#pprint(dimensions)
# print(dimensions.shape)



# In[29]:

import numpy
cmatrix = numpy.memmap("lsa.cmatrix", dtype='float64', mode='w+', shape=(len(unigrams),2*len(dimensions)))
print(cmatrix.shape)


# In[35]:

def populate_cmatrix(file_name, cmatrix, iunigrams, dimensions, window = 5):
     e = 0
     s = 0
     with open(file_name, encoding='utf-8', errors='ignore') as f:

        count = 0
        for index, line in enumerate(f):
            tokens = line.strip().split()
            posTokens = nltk.pos_tag(tokens)
            postokens = [i[0] for i in posTokens if i[1]=='VB' or i[1]=='VBD' or i[1]=='VBG' or i[1]=='VBN' or i[1]=='VBP' or i[1]=='VBZ']
            count+=1
            for indexj, token in enumerate(tokens):
                token = token.lower()
                lcontext = tokens[indexj - window:indexj]
                rcontext = tokens[indexj + 1:index + window]
                context = [tok.lower() for tok in lcontext + rcontext]
                
                try:
                    unigram_index = iunigrams[token]                    
                    for d in lcontext:
                        
                        if d in dimensions:
                            j = dimensions[d]
                            cmatrix[unigram_index][j] += 1
                    for d in rcontext:
                        
                        if d in dimensions:
                            j = dimensions[d]
                            cmatrix[unigram_index][1500+j] += 1
                            s += 1                          
                except:
                    e += 1
            
            
     print(e,s)
                
                


# In[36]:

from time import time
s = time()
populate_cmatrix(file_name, cmatrix, iunigrams, idimensions)
e = time()
print(e -s)


# In[37]:

#boy, sunday, eat, good, slowly, 100 
w1 = 'boy'
w2 = 'sunday'
w3 = 'eat'
w4 = 'good'
w5 = 'slowly'
w6 = '100'
id1 = iunigrams[w1]
id2 = iunigrams[w2]
id3 = iunigrams[w3]
id4 = iunigrams[w4]
id5 = iunigrams[w5]
id6 = iunigrams[w6]
print(id1, id2, id3, id4, id5)
v1 = cmatrix[id1]
v2 = cmatrix[id2]
v3 = cmatrix[id3]

print(v1, v2, v3)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import *
print(euclidean(v1, v2))
print(cosine(v1,v2))
a = ((v1.dot(v1))/(numpy.linalg.norm(v1)*numpy.linalg.norm(v1)))
print (1-a)


# In[38]:

from sklearn.decomposition import TruncatedSVD
s = time()
svd = TruncatedSVD(n_components=100, random_state=42)
svd.fit(cmatrix)
twod_cmatrix = svd.transform(cmatrix)
e = time()
print(e - s )


# In[39]:

v1_2d, v2_2d, v3_2d, v4_2d, v5_2d, v6_2d = twod_cmatrix[id1], twod_cmatrix[id2], twod_cmatrix[id3], twod_cmatrix[id4], twod_cmatrix[id5], twod_cmatrix[id6]
print ([runigrams[j] for i,j in enumerate([id1,id2,id3,id4,id5,id6])])
def getTen(vec):
    cosi = []
    for i in range(len(twod_cmatrix)):
        if numpy.linalg.norm(twod_cmatrix[i]) == 0:
            continue
        cosi.append((i,cosine(twod_cmatrix[i],vec)))
    cosi = sorted(cosi,key=lambda x: x[1])
    print (runigrams[cosi[0][0]],cosi[0][1], cosine(v1_2d,v1_2d))
    return [i[0] for i in cosi[1:11]]
for i in [v1_2d, v2_2d, v3_2d, v4_2d, v5_2d, v6_2d]:
    indx = getTen(i)
    print ([runigrams[i] for i in indx])


# In[ ]:

get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt
v1_2d = v1_2d / numpy.linalg.norm(v1_2d)
v2_2d = v2_2d / numpy.linalg.norm(v2_2d)
v3_2d = v3_2d / numpy.linalg.norm(v3_2d)
print ([v1_2d, v2_2d,v3_2d])
colors = ['r','b','g']
fig, axs = plt.subplots(1,1)
for i, x in enumerate([v1_2d, v2_2d,v3_2d]):
    a = plt.plot([0,x[0]],[0,x[1]],colors[i]+'-')
plt.show()


# In[10]:

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20,n_init=100,verbose=True)
temp = kmeans.fit(twod_cmatrix)
print (twod_cmatrix.shape)


# In[24]:

from collections import Counter
print (len(kmeans.cluster_centers_))


# In[ ]:




# In[ ]:




# In[ ]:




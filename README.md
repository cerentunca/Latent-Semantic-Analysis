# Latent-Semantic-Analysis
Python code which implements Latent Semantic Analysis
1. LSA

    1.1 Study the code and run it on given corpus (also attached).

    1.2 Try these following experiments

           1.2.1 Window sizes : 2, 5, 10

           1.2.2 Separate Windows : Maintain two halves of the dimensions : One for left context and one for the right context. Update the dimensions on the left context only for the words which occur on the left side of the word in consideration. Same goes for the right side. Do this for window sizes 2 & 5

          1.2.3 Use NLTK's POS tagger to tag 5000 of the sentences from the corpus. Now, while building the LSA matrix, consider only verbs in the context. (Choose your dimensions to be just verb-words).

          1.2.4 In all of the above choose initial dimension size to be 3000 and apply SVD to make it 100 dimensions. In this (1.2.4) perform the experiment (window=5) with SVD 200, 50, 10.

        1.3 For all the experiments above, report the top 10 similar words for the following list of words. (boy, sunday, eat, good, slowly, 100 (or any number)). Report your observations.

2. POS

    2.1 Take the case of SVD dimensions = 100 and window = 5 and cluster each word into 20 bins. Use Kmeans package from scipy/sklearn. 

   2.2 Report your findings on the clusters. Do they group words of the same pos tags in one cluster ?. What should be the cluster size ? 

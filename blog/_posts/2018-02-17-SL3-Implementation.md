---
layout: single
title: "Fake Hotel Review-3"
date: 2018-02-17
---

# Supervised Training Implementation 

## Preprocessing 
First I used ``gensim.utils.tokenize`` to parse the text, removing special characters and punctuation and splitting the text into an array of strings. I created bigrams from the text as it has been shown to increase accuracy with text-related tasks. 

```python
phrases = Phrases(sentence_stream)
bigram = Phraser(phrases)
sentence_stream = [bigram[s] for s in sentence_stream]
unlabelled_corpus = [bigram[s] for s in unlabelled_corpus]
```


I generated a list of stop words using an [original list of stopwords](https://github.com/bowenlow/ga_work/blob/master/final-project/input/stopwords.csv). Then I added on capitalized versions of stopwords and bigrams of stopwords, so that common phrases like *as if* can be grouped together. 

```python
new_stop = stop_words.stop.map(lambda x: str.capitalize(x))
all_stop_set = set(stop_words.stop.append(new_stop,ignore_index=True))

#generate combinations of all_stop_words, bigrams
#generate bigram
bigram_stops=[]
for a in all_stop_set:
    for b in all_stop_set:
        bigram_stop = a+"_"+b
        bigram_stops.append(bigram_stop)
all_stop_set = all_stop_set.union(bigram_stops)
```

I removed the stopwords from the text after creating the bigrams, to remove any remaining stopwords. 


## Creating Feature Vectors

### *VADER Sentiment Vectors* 
Using the [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text](https://github.com/cjhutto/vaderSentiment) Library by CJ Hutto and Eric Gilbert, I generate the sentiment vectors. There are 4 features created: compound, negative, neutral, positive. I use all four features, although they were correlated in some fashion. Extra features can always be removed later. Note that I used the raw, tokenized non-bigram text for VADER generation. It takes in a single string. 

```python
vader_sent = vader.polarity_scores(' '.join(raw_tokens))
``` 

### *LDA Topic Model Vectors*
Next, I generated the [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) topic models using ```gensim.models.LdaModel```. Generally the ```gensim``` library runs pretty efficiently. LDA modelling will capture the statistical collections of words that occur together; Perhaps fake reviews have similar topics, or higher rate of usage of certain words.  

```python
dictionary = gensim.corpora.Dictionary(sentence_stream)
corpus_bow = [dictionary.doc2bow(s) for s in sentence_stream]
num_topics = 100
chunksize = 400
passes = 5
model = gensim.models.LdaModel(corpus_bow, id2word=dictionary, num_topics=num_topics,alpha = 'auto',eta='auto',random_state=0, chunksize=chunksize, passes=passes)
```

### *Empath Topic Model Vectors*
LDA Vectors do not always create topics that are cleanly delineated; afterall it is just a statistical topic-word distribution that might not be very clear to humans. [Empath](https://hci.stanford.edu/publications/2016/ethan/empath-chi-2016.pdf), a topic model developed by a Stanford group, have modelled 198 human-curated topics such as *social media*, *war*, *technology*, etc. Researchers often use **LIWC** (Linguistic Inquiry and Word Count) features to analyze social media posts, however these features are proprietary, and according to the Stanford paper, they had very similar performance (average 0.90 Pearson correlation). Since this library is complete open-source, I used it. Note that raw, cleaned, non-bigram tokens were used. 

```python
from empath import Empath
lexicon = Empath()
lexicon_results = lexicon.analyze(' '.join(raw_tokens))
```

### *Doc2Vec Word Embeddings*
Word embeddings were created for each review. [This](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e) is a good introduction to the concept of word2vec, and doc2vec. Essentially I want to change text into a numerical representation of the context of the review. I used doc2vec as the reviews can be relatively long and it is natural to represent a single review as a document. The doc2vec parameters were as such, as advised as best practices in the Empath paper.
```python
doc2vec_model = gensim.models.doc2vec.Doc2Vec(dm=0, size=100,min_count=30, window=5,workers=cores, seed=8, negative=5)
```

I also used the unlabelled corpus as the training set - as doc2vec benefits from volume of data to learn the different possible contexts of each word. One must remember to tag each document in the corpus to be used
```python
for s in unlabelled_corpus:
	gensim.models.doc2vec.TaggedDocument(s,[ctr])
	ctr+=1
doc2vec_model.build_vocab(unlabelled_corpus)
doc2vec_model.train(unlabelled_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter)
corpus_vec = pd.DataFrame([doc2vec_model.infer_vector(s.words) for s in unlabelled_corpus])
```

Once all these feature vectors are ready, they are concatenated together and run through various different ```sklearn``` classifiers such as Guassian Naive Bayes, Decision Trees, SVMs, RandomForests, and XGBoost. After a *long* experimentation phase, I discovered that SVMs and XGBoost had the best consistent performance, and I ran GridSearch for hyperparameter discovery on XGBoost. 
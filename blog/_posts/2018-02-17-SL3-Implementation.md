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


First I generated a list of stop words using an [original list of stopwords](https://github.com/bowenlow/ga_work/blob/master/final-project/input/stopwords.csv). Then I added on capitalized versions of stopwords and bigrams of stopwords, so that common phrases like *as if* can be grouped together. 

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

### *Empath Topic Model Vectors*

### *Doc2Vec Word Embeddings*

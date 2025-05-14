#!/usr/bin/env python

import sys
import re
import nltk
from nltk import FreqDist, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

stop_words=set(stopwords.words("english"))

lem=WordNetLemmatizer()

def get_pos(tag):
    if re.match(r"^JJ",tag):
        return "a"
    elif re.match(r"^NN",tag) or re.match(r"^PRP",tag):
        return "n"
    elif re.match(r"^RB",tag):
        return "r"
    elif re.match(r"^VB",tag):
        return "v"
    return ""

if len(sys.argv)>1:
    file=sys.argv[1]
else:
    file=input("File name: ") or "text.txt"

with open(file) as f:
    text=f.read()
    sents=sent_tokenize(text)
    words=[word for sent in sents for word in word_tokenize(sent) if word.isalpha()]
    print("Number of words: %d" % len(words))
    tagged_words=nltk.pos_tag(words)
    lemmed_words=[lem.lemmatize(word[0],pos=get_pos(word[1])) if get_pos(word[1])!="" else lem.lemmatize(word[0]) for word in tagged_words]
   #print(lemmed_words)
    freqdist=FreqDist(lemmed_words)
    print("Twenty most common words: ")
    print(freqdist.most_common(20))
    meaningful_words=[word for word in lemmed_words if word.casefold() not in stop_words]
    print("Twenty most common words excluding stop words: ")
    freqdist2=FreqDist(meaningful_words)
    print(freqdist2.most_common(20))
    freqdist2.plot(20)
    plt.savefig("frequencydist")
    print("Collocations: ")
    print(nltk.Text(lemmed_words).collocations())

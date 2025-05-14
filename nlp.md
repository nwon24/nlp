# Introduction to NLP

Natural Language Processing (NLP) uses artifical intelligence to
understand and generate text in human languages. NLP is already
ubiquitious in everyday life, whether it be with voice-powered
assistants and online chatbots. What's exciting about NLP is its
potential to extend into the realm of sentiment analysis---that is,
its ability to analyse and recognise emotions, feelings, sarcasm,
mood, and other insights from largely unstructured text data. This, in
turn, constitutes a remarkable leap in data analysis.

## NLTK

Natural Language Toolkit (NLTK) is a Python library designed for
processing of English text. Here are some key concepts related to text
processing.
- Tokenising: breaking up text into its words
- Stop words: repeated words that don't add much to meaning and are
usually featured out (e.g., 'in', 'as', 'of')
- Stemming: reducing words to their root
- Lematizing: reducing words to their core meaning (e.g., best -> good)
- Chuncking: breaking up text into phrases---often the meaning of
individual words doesn't bear much relation to the phrase that it's
in

This week I began experimenting with the NLTK library, writing a simple script
to tokenize and lemmatize a text file before printing out the most common words,
both before and after taking out stop words.
A plot of the frequency distribution is also generated and saved to a file.

[Code](W1/textstats.py)

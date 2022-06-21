import pandas as pd
from gensim.models import FastText

# fastText builds on modern Mac OS and Linux distributions
# https://fasttext.cc/docs/en/support.html
# windows need additional settings see https://medium.com/@oleg.tarasov/building-fasttext-python-wrapper-from-source-under-windows-68e693a68cbb


# read data
emotion_keywords = pd.read_csv("word_tokens/emotion_keywords.csv")
emotion_abstracts = pd.read_csv("word_tokens/emotion_abstracts.csv")
cooperation_keywords = pd.read_csv("word_tokens/cooperation_keywords.csv")
cooperation_abstracts = pd.read_csv("word_tokens/cooperation_abstracts.csv")

# create list with words for training
emotion_keywords = emotion_keywords['word'].tolist()
emotion_abstracts = emotion_abstracts['word'].tolist()
cooperation_keywords = cooperation_keywords['word'].tolist()
cooperation_abstracts = cooperation_abstracts['word'].tolist()

# fasttext model (emotion_keywords)
model1 = FastText(vector_size=100, window=3, min_count=1)  # instantiate
model1.build_vocab(corpus_iterable=emotion_keywords)
model1.train(corpus_iterable=emotion_keywords, total_examples=len(emotion_keywords), epochs=10)
model1.most_similar("emotion")

# fasttext model (emotion_abstracts)
model2 = FastText(vector_size=100, window=3, min_count=1)  # instantiate
model2.build_vocab(corpus_iterable=emotion_keywords)
model2.train(corpus_iterable=emotion_keywords, total_examples=len(emotion_keywords), epochs=10)
model2.most_similar("emotion regulation")

# fasttext model (cooperation_keywords)
model3 = FastText(vector_size=100, window=3, min_count=1)  # instantiate
model3.build_vocab(corpus_iterable=emotion_keywords)
model3.train(corpus_iterable=emotion_keywords, total_examples=len(emotion_keywords), epochs=10)
model3.most_similar("cooperation")

# fasttext model (cooperation_abstracts)
model4 = FastText(vector_size=100, window=3, min_count=1)  # instantiate
model4.build_vocab(corpus_iterable=emotion_keywords)
model4.train(corpus_iterable=emotion_keywords, total_examples=len(emotion_keywords), epochs=10)
model4.most_similar("dilemma")

# model did not produce reliable vector representations 
# load model for futher training 
# model from https://fasttext.cc/docs/en/english-vectors.html
fb_model = FastText.load("cc.en.300.bin")
fb_model.build_vocab(emotion_keywords, update=True)  # Update the vocabulary
fb_model.train(emotion_keywords, total_examples=len(emotion_keywords), epochs=fb_model.epochs)
# large runtime and results in a memory error


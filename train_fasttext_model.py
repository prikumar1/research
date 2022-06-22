import pandas as pd
from gensim.models import FastText
from nltk.stem import WordNetLemmatizer

# fastText builds on modern Mac OS and Linux distributions
# https://fasttext.cc/docs/en/support.html
# windows need additional settings see https://medium.com/@oleg.tarasov/building-fasttext-python-wrapper-from-source-under-windows-68e693a68cbb

# read dataset
# first try with emotion dataset only 
emotion = pd.read_csv("recs_final.csv")

# remove empty rows
emotion_keywords = emotion["DE"].dropna()
emotion_abstracts = emotion["AB"].dropna()

# create function 
def tokenize(text):
    # convert to lowercase
    text = str(text).lower() 
    # tokenization 
    text = text.split()
    # lemmatization 
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

emotion_keywords = [tokenize(text) for text in emotion_keywords]
print(len(emotion_keywords)) # the same as the column 
emotion_abstracts = [tokenize(text) for text in emotion_abstracts]
print(len(emotion_abstracts)) # the same as the column 

# fasttext model (emotion_keywords)
model1 = FastText(vector_size=300, window=40, min_count=5, sample=1e-2, sg=1)
model1.build_vocab(corpus_iterable=emotion_keywords)
model1.train(corpus_iterable=emotion_keywords, total_examples=len(emotion_keywords), epochs=10)
print(model1.wv.most_similar("emotion", topn=10))
print(model1.wv.most_similar("adolescent", topn=10))
print(model1.wv.most_similar("child", topn=10))
print(model1.wv.most_similar("behaviour", topn=10))
print(model1.wv.most_similar("stress", topn=10))
print(model1.wv.most_similar("disorder", topn=10))
print(model1.wv.most_similar("gender", topn=10))
print(model1.wv.most_similar("boy", topn=10))
print(model1.wv.most_similar("hospital", topn=10))
print(model1.wv.most_similar("university", topn=10))

# fasttext model (emotion_abstracts)
model2 = FastText(vector_size=300, window=40, min_count=5, sample=1e-2, sg=1)
model2.build_vocab(corpus_iterable=emotion_abstracts)
model2.train(corpus_iterable=emotion_abstracts, total_examples=len(emotion_abstracts), epochs=10)
print(model2.wv.most_similar("emotion", topn=10))
print(model2.wv.most_similar("patients", topn=10))
print(model2.wv.most_similar("disorder", topn=10))
print(model2.wv.most_similar("treatment", topn=10))
print(model2.wv.most_similar("investigate", topn=10))
print(model2.wv.most_similar("child", topn=10))
print(model2.wv.most_similar("male", topn=10))
print(model2.wv.most_similar("sex", topn=10))
print(model2.wv.most_similar("female", topn=10))
print(model2.wv.most_similar("development", topn=10))
# -> model did not produce reliable vector representations 

# load model for futher training 
# model from https://fasttext.cc/docs/en/english-vectors.html
fb_model = FastText.load("cc.en.300.bin")
fb_model.build_vocab(emotion_keywords, update=True)  # Update the vocabulary
fb_model.train(emotion_keywords, total_examples=len(emotion_keywords), epochs=fb_model.epochs)
# -> large runtime and results in a memory error (requires large RAM/Memory/GPU)

# see script.Rmd --> load pre-trained word vectors trained using FastText and extract the relevant vector representations for the words
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import MeCab
import doc2vec
from gensim import models

model = models.Doc2Vec.load('doc2vec.model')

# 似た文章を探す
def search_similar_texts(words):
    x = model.infer_vector(words)
    most_similar_texts = model.docvecs.most_similar([x])
    for similar_text in most_similar_texts:
        print(similar_text[0])

# 似た単語を探す
def search_similar_words(words):
    for word in words:
        print()
        print(word + ':')
        for result in model.most_similar(positive=word, topn=10):
            print(result[0])

if __name__ == '__main__':
    print('文字列入力:')
    search_str = input()
    words = doc2vec.split_into_words(search_str).words
    search_similar_texts(words)
    search_similar_words(words)

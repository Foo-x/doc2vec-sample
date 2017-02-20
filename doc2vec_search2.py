#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import MeCab
import doc2vec
from gensim import models

DOC_PATH = './aozora/青空文庫/{0}/{1}.txt'

model = models.Doc2Vec.load('doc2vec.model')

# 似た文章を探す
def search_similar_texts(path):
    most_similar_texts = model.docvecs.most_similar(path)
    for similar_text in most_similar_texts:
        print(similar_text[0])

if __name__ == '__main__':
    print('作者:')
    author = input()
    print()
    print('作品:')
    work = input()
    print()
    search_similar_texts(DOC_PATH.format(author, work))

"""
The file preprocesses the data/train.txt, data/dev.txt and data/test.txt from sentiment classification task (English)
"""
from __future__ import print_function
import numpy as np
import gzip
import os
import util

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

def createMatrices(sentences, word2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']


    xMatrix = []
    unknownWordCount = 0
    wordCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        sentenceWordIdx = []

        for word in sentence:
            wordCount += 1

            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1

            sentenceWordIdx.append(wordIdx)

        xMatrix.append(sentenceWordIdx)


    print("Unknown tokens: %.2f%%" % (unknownWordCount/(float(wordCount))*100))
    return xMatrix

def readFile(filepath):
    sentences = []
    labels = []

    for line in open(filepath):
        splits = line.split()
        label = int(splits[0])
        words = splits[1:]

        labels.append(label)
        sentences.append(words)

    print(filepath, len(sentences), "sentences")

    return sentences, labels






# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #


def preprocess_data(embeddings_file, data_dir, pklf=None):

    #Train, Dev, and Test files
    files = [
        os.path.join(data_dir, 'train.txt'),
        os.path.join(data_dir, 'dev.txt'),
        os.path.join(data_dir, 'test.txt')
    ]



    trainDataset = readFile(files[0])
    devDataset = readFile(files[1])
    testDataset = readFile(files[2])


    # :: Compute which words are needed for the train/dev/test set ::
    words = {}
    for sentences, labels in [trainDataset, devDataset, testDataset]:
        for sentence in sentences:
            for token in sentence:
                words[token.lower()] = True


    # :: Load the pre-trained embeddings file ::
    wordEmbeddings, word2Idx = util.load_embeddings_matrix(embeddings_file, words)

    print("Embeddings shape: ", wordEmbeddings.shape)
    print("Len words: ", len(words))



    # :: Create matrices ::
    train_matrix = createMatrices(trainDataset[0], word2Idx)
    dev_matrix = createMatrices(devDataset[0], word2Idx)
    test_matrix = createMatrices(testDataset[0], word2Idx)


    data = {
        'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
        'train': {'sentences': train_matrix, 'labels': trainDataset[1]},
        'dev':   {'sentences': dev_matrix, 'labels': devDataset[1]},
        'test':  {'sentences': test_matrix, 'labels': testDataset[1]}
        }


    if pklf:
        f = gzip.open(pklf, 'wb')
        pkl.dump(data, f)
        f.close()
        print("Pickled preprocessed data to %s" % pklf)

    return data
    

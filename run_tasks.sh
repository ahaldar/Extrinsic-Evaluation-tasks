#!/bin/bash

PY=/usr/bin/python
EMBEDDINGS=test-embeddings.txt

echo "=== Data management ==="

if [ ! -d snli/dataset ]; then
    mkdir snli/dataset
fi
if [ ! -d snli/dataset/snli_1.0 ]; then
    cd snli/dataset
    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip snli_1.0.zip
    cd ../../
fi

echo "=== Relation extraction ==="
${PY} -m Relation_extraction.train_cnn \
    ${EMBEDDINGS} \
    Relation_extraction/dataset

echo "=== Sentence polarity classification ==="
${PY} -m sentence_polarity_classification.train \
    ${EMBEDDINGS} \
    sentence_polarity_classification/data

echo "=== Sentiment classification ==="
${PY} -m sentiment_classification.train \
    ${EMBEDDINGS}

echo "=== Subjectivity classification ==="
${PY} -m subjectivity_classification.cnn \
    ${EMBEDDINGS} \
    subjectivity_classification/data

echo "=== SNLI ==="
${PY} -m snli.train \
    ${EMBEDDINGS} \
    snli/dataset/snli_1.0

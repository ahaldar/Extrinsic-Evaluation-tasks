# Extrinsic-Evaluation-tasks

Fork of [shashwath94/Extrinsic-Evaluation-tasks](https://github.com/shashwath94/Extrinsic-Evaluation-tasks), with some cleaning to support running all tasks on an arbitrary set of embeddings more easily.

Original repository assembled for Rogers et al. (2018) "[What's in Your Embedding, And How It Predicts Task Performance](http://aclweb.org/anthology/C18-1228)".

This version of repository assembled and released due to usage in Whitaker et al. (2019) "[Characterizing the impact of geometric properties of word embeddings on task performance](https://arxiv.org/abs/1904.04866)".

## Specific details for each task

_References pending_

## References

Bib entry for Rogers et al (2018):
```
@inproceedings{C18-1228,
  title = "What{'}s in Your Embedding, And How It Predicts Task Performance",
  author = "Rogers, Anna  and Hosur Ananthakrishna, Shashwath  and Rumshisky, Anna",
  booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
  month = aug,
  year = "2018",
  address = "Santa Fe, New Mexico, USA",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/C18-1228",
  pages = "2690--2703",
}
```

Bib entry for Whitaker et al (2019):
```
@inproceedings{Whitaker2019,
  title = "Characterizing the impact of geometric properties of word embeddings on task performance",
  author = "Whitaker, Brendan and Newman-Griffis, Denis and Haldar, Aparajita and Ferhatosmanoglu, Hakan and Fosler-Lussier, Eric",
  booktitle = "Proceedings of the 3rd Workshop on Evaluating Vector Space Representations for NLP",
  month = jun,
  year = "2019",
  address = "Minneapolis, Minnesota, USA",
  publisher = "Association for Computational Linguistics"
}
```

## Old README

For each task, run `preprocess.py` to load the preprocessed version of the dataset.
To train the model, run `train.py`

A pretrained word embedding text file is needed where every line has a word string followed by a space and the embedding vector.
For example, `acrobat 0.6056159735 -0.1367940009 -0.0936380029 0.8406270146 0.2641879916 0.4209069908 0.0607739985 0.5985950232 -1.1451450586 -0.8666719794 -0.5021889806 0.4398249984 0.9671009779 0.7413169742 -0.0954160020 -1.1526989937 -0.3915260136 -0.1520590037 0.0893440023 -0.2578850091 -0.6204599738 -0.8789629936 0.3581469953 0.5509790182 0.1234730035`

Data for NLI task can be found [here](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

For the sequence labeling tasks(POS, NER and chunking), please refer to [this repo](https://github.com/shashwath94/Sequence-Labeling)

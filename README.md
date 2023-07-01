# Quran Verse Chunker

*Join us on a new journey! Visit the [Corpus 2.0 upgrade project](https://github.com/kaisdukes/quranic-corpus) for new work on the Quranic Arabic Corpus.*

## What’s in this Repo?

A data preprocessor for the new [Quranic Treebank](https://qurancorpus.app/treebank/2:258). Divides longer verses into smaller chunks.

To work with this codebase, you will need a strong background in Artificial Intelligence applied to Quranic Research, specifically in the fields of Computational Linguistics and Natural Language Processing (NLP).

## Why Do We Need This?

Large portions of the Quran contain lengthy verses. The Quranic Treebank is designed primarily as an educational resource, allowing users of the corpus to gain deeper linguistic understanding of the Classical Arabic language of the Quran through side-by-side comparison with *i’rāb* (إعراب), traditional linguistic analysis. Dependency graphs are kept intentionally short, for easier display on mobile devices. Larger syntactic structures that cross graphs are linked together through reference nodes.

To construct the treebank, we need to first perform verse chunking. There are several ways this could be done, but one possibility is to train a machine learning model using four sources of data:

**Existing chunk boundaries:** The chunks implied by the existing dependency graphs in the treebank.

**Reference grammar alignment:** The breakdown of verses into word groups in the reference grammar used to construct the treebank, Salih’s *al-I’rāb al-Mufassal*. In principle, this could be a strong choice for training the model as the treebank was initially chunked to support easier alignment and cross-referencing with this reference work.

**Pause marks:** Although the Classical Arabic Uthmani script of the Quran doesn’t contain modern punctuation like full stops or commas, it does contain [pause marks](https://corpus.quran.com/documentation/pausemarks.jsp) (to support *waqf* and *tajweed*), which may aid in chunking.

**Punctuation from translations:** The Quranic Arabic Corpus has word-aligned translations into English, which often include punctuation. Using this data may also help boost the accuracy of the chunker.

Because the evaluation step needs to test against the treebank, it makes sense to include the existing implied chunk boundaries as part of the training dataset. Other data sources are included to test how they might boost accuracy. Choosing just one signal, like *waqf* marks might not be optimal.

## What’s in the Data File?

A ‘word’ in the Quran isn't easily defined, due to the rich morphology of Classical Arabic. The Quranic Arabic Corpus uses the terminology ‘segment’ to denote a morphological segment and ‘token’ to denote a whitespace separated token.

The [quranic-treebank-0.4-chunks.tsv](https://github.com/kaisdukes/quran-verse-chunker/tree/main/data) file has one row per token with 9 columns:

* Chapter number
* Verse number
* Token number
* The arabic form of the token (in Unicode)
* The [POS tag](https://corpus.quran.com/documentation/tagset.jsp) of the token’s stem (note: to support initial experimentation, this is a simplification of the full morphological data available in the corpus, which includes a rich set of [features tags](https://corpus.quran.com/documentation/morphologicalfeatures.jsp) for each segment, not only stems)
* The world-aligned english translation, including punctuation marks such as full stops
* The [pause mark](https://corpus.quran.com/documentation/pausemarks.jsp) (*waqf*) associated with the token
* A binary flag indicating if the token is at the end of an word group in the corresponding *i’rāb* (إعراب) in the reference grammar *al-I’rāb al-Mufassal*
* A binary flag indicating if the token is at the end of a dependency graph. This value is the expected output of the chunker.

## Getting Started

This project uses [Poetry](https://python-poetry.org) to manage package dependencies.

First, clone the repository:

```
git clone https://github.com/kaisdukes/quran-verse-chunker.git
cd quran-verse-chunker
```

Install Poetry using [Homebrew](https://brew.sh):

```
brew install poetry
```

Next, install project dependencies:

```
poetry install
```

All dependencies, such as [pandas](https://pandas.pydata.org), are installed in the virtual environment.

Use the Poetry shell:

```
poetry shell
```

Verify the dataset:

```
python tests/data_test.py
```
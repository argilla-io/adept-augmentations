# adept-augmentations

A Python library aimed at adeptly augmenting NLP training data.

## Usage

### Datasets

```python
from datasets import load_dataset

from adept_augmentations import EntitySwapAugmenter

dataset = load_dataset("conll2003", split="train[:3]")
augmenter = EntitySwapAugmenter(dataset)
aug_dataset = augmenter.augment(N=4)

for entry in aug_dataset["tokens"]:
    print(entry)

# ['EU', 'rejects', 'British', 'call', 'to', 'boycott', 'British', 'lamb', '.']
# ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'German', 'lamb', '.']
# ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
# ['Peter', 'Blackburn']
# ['BRUSSELS', '1996-08-22']
```

### spaCy

```python
import spacy
from spacy.tokens import DocBin

from adept_augmentations import EntitySwapAugmenter

nlp = spacy.load("en_core_web_sm")

# Create some example training data
TRAIN_DATA = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "Microsoft acquires GitHub for $7.5 billion",
]
docs = nlp.pipe(TRAIN_DATA)

# Create a new DocBin
doc_bin = DocBin(docs=docs)

doc_bin = EntitySwapAugmenter(doc_bin).augment(4)
for doc in doc_bin.get_docs(nlp.vocab):
    print(doc.text)

# GitHub is looking at buying U.K. startup for $ 7.5 billion
# Microsoft is looking at buying U.K. startup for $ 1 billion
# Microsoft is looking at buying U.K. startup for $ 7.5 billion
# GitHub is looking at buying U.K. startup for $ 1 billion
# Microsoft acquires Apple for $ 7.5 billion
# Apple acquires Microsoft for $ 1 billion
# Microsoft acquires Microsoft for $ 7.5 billion
# GitHub acquires GitHub for $ 1 billion
```

## Implemented Augmenters

- [X] `EntitySwapAugmenter`
- [ ] `KnowledgeBaseSwapAugmenter`
- [ ] `CoreferenceSwapAugmenter`
- [ ] `SyntaticTreeSwapAugmenter`

## Potential integrations

Potentially, we can look into integrations of other augmentations packages that do not preserve gold standard knowledge. Good sources for inspiration are:

- <https://github.com/KennethEnevoldsen/augmenty>
  - <https://kennethenevoldsen.github.io/augmenty/tutorials/introduction.html>
- <https://github.com/QData/TextAttack>
- <https://github.com/infinitylogesh/mutate>

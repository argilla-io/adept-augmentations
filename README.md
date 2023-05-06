# Adept Augmentations

Welcome to Adept Augmentation, the ultimate solution for those struggling with few shot Named Entity Recognition (NER) tasks!

Adept Augmentation is a Python package that provides data augmentation functionalities for NER training data using the `spacy` and `datasets` packages. With Adept Augmentation, you can easily generate more labeled data for your few shot NER task and improve your model performance.

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

## Potential performance gains
Data augmentation can significantly improve model performance in low-data scenarios.
To showcase this, we trained a [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) NER model on
the 50, 100, 200, 400 and 800 first [CoNLL03](https://huggingface.co/datasets/conll2003) training samples.

The augmented dataset is generated like so:
```python
# Select N (50, 100, 200, 400 or 800) samples from the gold training dataset
train_dataset = dataset["train"].select(range(N))

# Generate augmented dataset, with 4 * N samples
augmented_dataset = Augmenter(train_dataset).augment(N=4)

# Combine the original with the augmented to produce the full dataset
# to produce a dataset 5 times as big as the original
train_dataset = concatenate_datasets([augmented_dataset, train_dataset])
```

Note that the baseline uses 5 epochs. This way, the training time and steps are identical between the two experiments. All scenarios are executed 5 times,
and we report means and standard errors.

|       | Original - 5 Epochs | Augmented - 1 Epoch |
|-------|--|--|
| N=50  | 0.387 ± 0.042 F1 | **0.484 ± 0.054 F1** |
| N=100 | 0.585 ± 0.070 F1 | **0.663 ± 0.038 F1** |
| N=200 | 0.717 ± 0.053 F1 | **0.757 ± 0.025 F1** |
| N=400 | 0.816 ± 0.017 F1 | **0.826 ± 0.011 F1** |
| N=800 | 0.859 ± 0.004 F1 | **0.862 ± 0.002 F1** |

(Note: These results are not optimized and do not indicate maximum performances with SpanMarker.)

From these results, it is clear that performing data augmentation using `adept_augmentations` can heavily improve performance in low-data settings.

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

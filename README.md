# Adept Augmentations

Welcome to Adept Augmentations, which can be used for creating additional data in Few Shot Named Entity Recognition (NER) settings!

Adept Augmentation is a Python package that provides data augmentation functionalities for NER training data using the `spacy` and `datasets` packages. Currently, we support one augmentor `EntitySwapAugmenter`, however, we plan on [adding some more](#implemented-augmenters).

`EntitySwapAugmenter` takes either a `datasets.Dataset` or a `spacy.tokens.DocBin`. Additionally, it is optional to provide a set of `labels`. It initially creates a knowledge base of entities belonging to a certain label. When running `augmenter.augment()` for `N` runs, it then creates `N` new sentences with random swaps of the original entities with an entity of the same corresponding label from the knowledge base.

For example, assuming that we have knowledge base for `PERSONS`, `LOCATIONS` and `PRODUCTS`. We can then create additional data for the sentence "Momofuko Ando created instant noodles in Osaka." using `augmenter.augment(N=2)`, resulting in "David created instant noodles in Madrid." or "Tom created Adept Augmentations in the Netherlands".

Adept Augmentation works for NER labels using the IOB, IOB2, BIOES and BILUO tagging schemes, as well as labels not following any tagging scheme.

## Usage

### Datasets

```python
from adept_augmentations import EntitySwapAugmenter
from datasets import load_dataset

golden_dataset = load_dataset("conll2003", split="train[:3]")
augmenter = EntitySwapAugmenter(golden_dataset)
augmented_dataset = augmenter.augment(N=4)

for entry in augmented_dataset["tokens"]:
    print(entry)

# ['EU', 'rejects', 'British', 'call', 'to', 'boycott', 'British', 'lamb', '.']
# ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'German', 'lamb', '.']
# ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
# ['Peter', 'Blackburn']
# ['BRUSSELS', '1996-08-22']
```

### spaCy

```python
from adept_augmentations import EntitySwapAugmenter
import spacy
from spacy.tokens import Doc, DocBin

nlp = spacy.blank("en")

# Create some example golden data
example_data = [
    ("Apple is looking at buying U.K. startup for $1 billion", [(0, 5, "ORG"), (27, 31, "LOC"), (44, 54, "MONEY")]),
    ("Microsoft acquires GitHub for $7.5 billion", [(0, 9, "ORG"), (19, 25, "ORG"), (30, 42, "MONEY")]),
]

# Create a new DocBin
nlp = spacy.blank("en")
docs = []
for entry in example_data:
    doc = Doc(nlp.vocab, words=entry[0].split())
    doc.ents = [doc.char_span(ent[0], ent[1], label=ent[2]) for ent in entry[1]]
    docs.append(doc)
golden_dataset = DocBin(docs=docs)

# Augment Data
augmented_dataset = EntitySwapAugmenter(golden_dataset).augment(4)
for doc in augmented_dataset.get_docs(nlp.vocab):
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
golden_dataset = dataset["train"].select(range(N))

# Generate augmented dataset, with 4 * N samples
augmented_dataset = Augmenter(golden_dataset).augment(N=4)

# Combine the original with the augmented to produce the full dataset
# to produce a dataset 5 times as big as the original
augmented_dataset = concatenate_datasets([augmented_dataset, golden_dataset])
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
- [ ] `SentenceCropAugmenter`

## Potential integrations

Potentially, we can look into integrations of other augmentations packages that do not preserve gold standard knowledge. Good sources for inspiration are:

- <https://github.com/KennethEnevoldsen/augmenty>
  - <https://kennethenevoldsen.github.io/augmenty/tutorials/introduction.html>
- <https://github.com/QData/TextAttack>
- <https://github.com/infinitylogesh/mutate>

## Logo Attribution

Augmented Reality by KΛPKLΛM from <a href="https://thenounproject.com/browse/icons/term/augmented-reality/" target="_blank" title="Augmented Reality Icons">Noun Project</a>


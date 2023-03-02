# adept-augmentations

A Python library aimed at dissecting and augmenting NLP training data.

IMO, we can separate the idea in 3 components:

1. analyze
2. report (lowest priority)
3. augment

# TO-DO

- [] simple POC of efficiency NER replacement via `augmenty`
  - create NER(word,tag) KB from n=[2,4,8,16,32] samples from arbitrary datasets
  - use `augmenty.entity_augmenter` with KB to create additional data with `level=1`, i.e., 100% replacement
  - check the impact in size for n=[2,4,8,16,32] -> m=[?,?,?,?,?]
  - log data into `argilla` and easily evaluate with `autotrain`
- [] basic Analyzer for IOB2 tags [https://huggingface.co/datasets/conll2003]

# Components

## Analyzer

We should define a set of analyzers that are built on top of a `AnalyzerBase`, which output an intermediary representation of the `TokenClasses` (POS, NER, CoRef) based on different labelling schemas. For me, these augmentations can either be applied to a gold standard or just inferred knowledge from a pre-trained model, e.g., POS and DEP structures.

### Sub-analyzers

- [] `AnalyzerBase`
- [] `AnalyzerIOB`
- [] `AnalyzerIOB2`
- [] `AnalyzerBIO`
- [] `AnalyzerCharSpan`
- [] `AnalyzerTokenSpan`

## Augmenter

We should define a set of augmentation recipes for each type of structural changes we might want to apply, while preserving gold standard knowledge w.r.t. target tags required for training.

### Sub-augmenters

- [] `AugmenterBase`
- [] `AugmenterNER`
- [] `AugmenterPOS`
- [] `AugmenterDEP`
- [] `AugmenterCOREF`

### integrations

Potentially, we can look into integrations of other augmentations packages that do not preserve gold standard knowledge. For me, `augmenty` works nicely, but I am unsure about the value of these augmentations w.r.t. getting valuable training data with simpel word/character swaps.

- <https://github.com/KennethEnevoldsen/augmenty>
  - <https://kennethenevoldsen.github.io/augmenty/tutorials/introduction.html>
- <https://github.com/QData/TextAttack>
- <https://github.com/infinitylogesh/mutate>

## Reporter

Lowest priority, but it could be cool to be able to visualize this using `datapane` or other reporting tools.

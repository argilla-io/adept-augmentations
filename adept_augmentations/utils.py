from datasets import ClassLabel, Dataset, Features, Sequence, Value
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab


def convert_docbin_to_dataset(doc_bin: DocBin, labels=None) -> Dataset:
    """
    This function converts a spaCy DocBin object into a dataset, with optional labels.

    Args:
      doc_bin (DocBin): The `doc_bin` parameter is a `DocBin` object, which is a container for storing
    spaCy `Doc` objects in a binary format. This is often used for efficient serialization and
    deserialization of large amounts of text data.
      labels: The `labels` parameter is an optional argument that can be passed to the function
    `convert_docbin_to_dataset()`. It is used to specify the labels for the data in the `DocBin` object.
    If `labels` is not provided, the function will assume that the labels are already present
    """
    vocab = Vocab()

    unique_labels = set()
    if labels is None:
        docs = doc_bin.get_docs(vocab)
        for doc in docs:
            for ent in doc.ents:
                unique_labels.add(ent.label_)
        labels = list(unique_labels)

    labels = ["O"] + list(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    features = {
        "tokens": Sequence(feature=Value(dtype="string")),
        "ner_tags": Sequence(feature=ClassLabel(names=labels)),
    }

    datasets_dict = {"tokens": [], "ner_tags": []}
    for doc in doc_bin.get_docs(vocab):
        datasets_dict["tokens"].append([token.text for token in doc])
        datasets_dict["ner_tags"].append([label2id.get(token.ent_type_, "O") for token in doc])

    converted_dataset = Dataset.from_dict(mapping=datasets_dict, features=Features(features))

    return converted_dataset


def convert_dataset_to_docbin(dataset: Dataset) -> DocBin:
    vocab = Vocab()
    label_list = dataset.features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}

    def get_doc_from_entry(tokens: list, ner_tags: list) -> Doc:
        """
        This function takes in a list of tokens and named entity recognition tags, and returns a spaCy
        Doc object with BILUO labels for the entity types.

        Args:
          tokens (list): a list of strings representing the words in a sentence
          ner_tags (list): The `ner_tags` parameter is a list of named entity recognition tags assigned
        to each token in the `tokens` list. These tags indicate whether a token is part of a named
        entity and, if so, what type of entity it is (e.g. person, organization, location, etc.).

        Returns:
          a spaCy `Doc` object with the input tokens and named entity recognition (NER) tags converted
        to BILUO labels. The function also removes any entities with the label "O" (which means they
        were not recognized as named entities).
        """

        # Define the BILUO labels for the entity types
        biluo_labels = []
        prev_label = None
        for label in ner_tags:
            if label != "O":
                label = id2label[label]
                if prev_label != label:
                    biluo_labels.append("B-" + str(label))
                else:
                    biluo_labels.append("I-" + str(label))
            else:
                biluo_labels.append(None)
            prev_label = label
        doc = Doc(vocab, words=tokens, ents=biluo_labels)
        doc.ents = [ent for ent in doc.ents if ent.label_ != "O"]
        return doc

    doc_bin = DocBin()
    for entry in dataset:
        doc_bin.add(get_doc_from_entry(entry["tokens"], entry["ner_tags"]))

    return doc_bin

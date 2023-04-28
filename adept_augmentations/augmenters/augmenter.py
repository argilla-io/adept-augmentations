import random
from collections import defaultdict
from enum import Enum, auto
from typing import List, Optional, Set, Union

from datasets import Dataset
from spacy.tokens import DocBin

from adept_augmentations.augmenters.constants import Entity
from adept_augmentations.augmenters.extractors import (
    EntityExtractorBILOU,
    EntityExtractorBIOES,
    EntityExtractorIOB,
    EntityExtractorNoScheme,
)
from adept_augmentations.utils import (
    convert_dataset_to_docbin,
    convert_docbin_to_dataset,
)


class LabelScheme(Enum):
    NONE = auto()
    IOB2 = auto()  # Works for IOB, too
    BIOES = auto()
    BILOU = auto()

    @staticmethod
    def are_labels_schemed(labels) -> bool:
        """True if all labels are strings matching one of the two following rules:

        * `label == "O"`
        * `label[0] in "BIESLU"` and `label[1] == "-"`, e.g. in `"I-LOC"`

        We ensure that the first index is in `"BIELSU"` because of these definitions:
        * `"B"` for `"begin"`
        * `"I"` for `"in"`
        * `"E"` for `"end"`
        * `"L"` for `"last"`
        * `"S"` for `"singular"`
        * `"U"` for `"unit"`

        Args:
            id2label (Dict[int, str]): Dictionary of label ids to label strings.

        Returns:
            bool: True if it seems like a labeling scheme is used.
        """
        return all(label == "O" or (len(label) > 2 and label[0] in "BIELSU" and label[1] == "-") for label in labels)

    def get_scheme_tags(labels) -> Set[str]:
        return set(label[0] for label in labels)

    @classmethod
    def from_labels(cls, labels: List[str]):
        if not cls.are_labels_schemed(labels):
            return cls.NONE, EntityExtractorNoScheme(labels)

        tags = cls.get_scheme_tags(labels)
        if tags == set("IOB"):
            return cls.IOB2, EntityExtractorIOB(labels)
        if tags == set("BIOES"):
            return cls.BIOES, EntityExtractorBIOES(labels)
        if tags == set("BILOU"):
            return cls.BILOU, EntityExtractorBILOU(labels)
        raise NotImplementedError(f"The detected labeling scheme with tags {tags!r} has not been implemented.")


class EntitySwapAugmenter:
    def __init__(
        self, dataset: Union[Dataset, DocBin], labels: Optional[List[str]] = None, label_column: str = "ner_tags"
    ) -> None:
        self.dataset_type = type(dataset)
        if self.dataset_type == DocBin:
            dataset = convert_docbin_to_dataset(dataset, labels)
        elif self.dataset_type != Dataset:
            raise TypeError("dataset must be either a `datasets.Dataset` or a `spacy.tokens.DocBin`.")

        self.dataset = dataset
        if labels is None:
            # TODO: This won't always work
            labels = dataset.features[label_column].feature.names
        self.labels = labels
        self.label_column = label_column

        self.label_scheme, self.entity_extractor = LabelScheme.from_labels(labels)

        # TODO: Require `tokens` in dataset
        # TODO: Ensure that "entities" doesn't already exist in dataset
        self.knowledge_base = defaultdict(set)
        self.dataset = self.dataset.map(
            self.extract_entities, input_columns=["tokens", label_column], load_from_cache_file=False
        )

    def augment(self, N: int = 4, deduplicate: bool = True) -> Union[Dataset, DocBin]:
        # TODO: Rename N, perhaps to "runs"?
        # N is the number of times we reuse every sentence
        augmented_dataset = self.dataset.map(
            self.replace_entities,
            input_columns=["tokens", self.label_column, "entities"],
            remove_columns=self.dataset.column_names,
            load_from_cache_file=False,
            fn_kwargs={"N": N, "deduplicate": deduplicate},
            batched=True,
        )
        if self.dataset_type == DocBin:
            return convert_dataset_to_docbin(augmented_dataset)
        else:
            return augmented_dataset

    def extract_entities(self, tokens: List[str], labels: List[int]):
        entities = list(self.entity_extractor(labels))
        for label, start, end in entities:
            entity = tuple(tokens[start:end])
            # TODO: Check why sometimes the entity has length 0
            if entity:
                self.knowledge_base[label].add(entity)
        return {"tokens": tokens, self.label_column: labels, "entities": entities}

    def replace_entities(
        self,
        batch_tokens: List[str],
        batch_labels: List[int],
        batch_entities: List[Entity],
        N: int = 4,
        deduplicate: bool = True,
    ):
        # TODO: Convert labels correctly for IOB, etc.
        batch = {
            "tokens": [],
            self.label_column: [],
        }

        for tokens, labels, entities in zip(batch_tokens, batch_labels, batch_entities):
            seen_texts = set()
            for _ in range(N):
                tokens_copy = tokens[::]
                labels_copy = labels[::]
                """
                Two variations exist here:
                * Using the for-loop we can replace all entities, and
                * using the random.choice we can replace a random one.
                """
                for label, start, end in entities[::-1]:
                    # if entities:
                    #     label, start, end = random.choice(entities)
                    entity_tokens = random.choice(tuple(self.knowledge_base[label]))
                    tokens_copy[start:end] = entity_tokens
                    labels_copy[start:end] = self.entity_extractor.reduced_label_id_to_id(label, len(entity_tokens))
                assert len(tokens_copy) == len(labels_copy)
                tokens_copy_str = " ".join(tokens_copy)
                if tokens_copy_str not in seen_texts:
                    batch["tokens"].append(tokens_copy)
                    batch[self.label_column].append(labels_copy)
                    if deduplicate:
                        seen_texts.add(tokens_copy_str)
        return batch

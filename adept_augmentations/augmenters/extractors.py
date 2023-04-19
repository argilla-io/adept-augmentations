from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Set, Tuple

from adept_augmentations.augmenters.constants import Entity


class EntityExtractor(ABC):
    """Class to convert NER training data into a common format used in the SpanMarkerModel.

    The common format involves tokenized texts and labels as Entity instances.
    """

    def __init__(self, labels: List[str]) -> None:
        super().__init__()
        self.labels = labels

    @abstractmethod
    def __call__(self, ner_tags: List[int]) -> Dict[str, List[Any]]:
        return

    def group_label_ids_by_tag(self) -> Dict[str, Set]:
        grouped = defaultdict(set)
        for label_id, label in enumerate(self.labels):
            grouped[label[0]].add(label_id)
        return dict(grouped)


class EntityExtractorScheme(EntityExtractor):
    def __init__(self, labels: List[str]) -> None:
        super().__init__(labels)
        self.label_ids_by_tag = self.group_label_ids_by_tag()
        self.start_ids = set()
        self.end_ids = set()

        reduced_labels = {label[2:] for label in self.labels if label != "O"}
        reduced_labels = ["O"] + sorted(reduced_labels)
        self.id2reduced_id = {
            _id: reduced_labels.index(label[2:] if label != "O" else label) for _id, label in enumerate(self.labels)
        }

    def __call__(self, ner_tags: List[int]) -> Iterator[Entity]:
        """Assumes a correct IOB or IOB2 annotation scheme"""
        start_idx = None
        reduced_label_id = None
        for idx, label_id in enumerate(ner_tags):
            # End of an entity
            if start_idx is not None and label_id in self.end_ids:
                yield (reduced_label_id, start_idx, idx)
                start_idx = None

            # Start of an entity
            if start_idx is None and label_id in self.start_ids:
                # compute the schemeless label ID
                reduced_label_id = self.id2reduced_id[label_id]
                start_idx = idx

        if start_idx is not None:
            yield (reduced_label_id, start_idx, idx)


class EntityExtractorIOB(EntityExtractorScheme):
    def __init__(self, labels: List[str]) -> None:
        super().__init__(labels)
        # Support for IOB2 and IOB, respectively:
        self.start_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["I"]
        self.end_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["O"]


class EntityExtractorBIOES(EntityExtractorScheme):
    def __init__(self, labels: List[str]) -> None:
        super().__init__(labels)
        self.start_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["S"]
        self.end_ids = self.label_ids_by_tag["B"] | self.label_ids_by_tag["O"] | self.label_ids_by_tag["S"]


class EntityExtractorBILOU(EntityExtractorScheme):
    def __init__(self, labels: List[str]) -> None:
        super().__init__(labels)
        self.start_ids = self.label_ids_by_tag["B"] & self.label_ids_by_tag["U"]
        self.end_ids = self.label_ids_by_tag["B"] & self.label_ids_by_tag["O"] & self.label_ids_by_tag["U"]


class EntityExtractorNoScheme(EntityExtractor):
    def __init__(self, labels: List[str]) -> None:
        super().__init__(labels)
        self.outside_id = labels.index("O")

    def __call__(self, ner_tags: List[int]) -> Iterator[Entity]:
        start_idx = None
        entity_label_id = None
        for idx, label_id in enumerate(ner_tags):
            # End of an entity
            if start_idx is not None and label_id != entity_label_id:
                yield (entity_label_id, start_idx, idx)
                start_idx = None

            # Start of an entity
            if start_idx is None and label_id != self.outside_id:
                entity_label_id = label_id
                start_idx = idx

        if start_idx is not None:
            yield (entity_label_id, start_idx, idx)
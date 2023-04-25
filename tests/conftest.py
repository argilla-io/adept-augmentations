from typing import List
from datasets import ClassLabel, Sequence, load_dataset, Dataset
import datasets
import pytest
from spacy.training import iob_to_biluo

from tests.constants import CONLL_LABELS, BILOU_CONLL_LABELS


def pytest_sessionstart(session) -> None:
    # Disable caching for testing only to ensure that we're actually recomputing things
    datasets.disable_caching()


@pytest.fixture(scope="session")
def conll03_tiny() -> Dataset:
    # an IOB2 dataset
    return load_dataset("conll2003", split="train[:100]")


@pytest.fixture(scope="session")
def bilou_conll03_tiny(conll03_tiny: Dataset) -> Dataset:
    def iob_to_bilou(iob_label_ids: List[int]) -> List[int]:
        iob_labels = [CONLL_LABELS[iob_label_id] for iob_label_id in iob_label_ids]
        bilou_labels = iob_to_biluo(iob_labels)
        bilou_label_ids = [BILOU_CONLL_LABELS.index(bilou_label) for bilou_label in bilou_labels]
        return {"ner_tags": bilou_label_ids}

    features = conll03_tiny.features.copy()
    features["ner_tags"] = Sequence(feature=ClassLabel(names=BILOU_CONLL_LABELS))
    bilou_conll03 = conll03_tiny.cast(features)
    return bilou_conll03.map(iob_to_bilou, input_columns="ner_tags")


@pytest.fixture(scope="session")
def fewnerd_coarse_tiny() -> Dataset:
    # an unschemed dataset
    return load_dataset("DFKI-SLT/few-nerd", "supervised", split="train[:100]")


@pytest.fixture(scope="session")
def fabner_tiny() -> Dataset:
    # a BIOES dataset
    return load_dataset("DFKI-SLT/fabner", split="train[:100]")

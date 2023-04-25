from typing import List, Optional
import pytest
from adept_augmentations import Augmenter
from tests.constants import CONLL_LABELS, FEWNERD_COARSE_LABELS, FABNER_LABELS, BILOU_CONLL_LABELS


@pytest.mark.parametrize("N", (2,))
@pytest.mark.parametrize(
    ("dataset_fixture", "labels"),
    (
        ("conll03_tiny", CONLL_LABELS), # <- IOB2
        ("conll03_tiny", None),
        ("fewnerd_coarse_tiny", FEWNERD_COARSE_LABELS), # <- Unschemed
        ("fewnerd_coarse_tiny", None),
        ("fabner_tiny", FABNER_LABELS), # <- BIOES
        ("fabner_tiny", None),
        ("bilou_conll03_tiny", BILOU_CONLL_LABELS), # <- BIOES
        ("bilou_conll03_tiny", None),
    ),
)
def test_augmenter(N: int, dataset_fixture: str, labels: Optional[List[str]], request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(dataset_fixture)
    augmenter = Augmenter(dataset, labels=labels)
    augmented_ds = augmenter.augment(N=N)
    assert len(augmented_ds) == len(dataset) * N


def test_augmenter_zero(conll03_tiny) -> None:
    augmenter = Augmenter(conll03_tiny, labels=CONLL_LABELS)
    augmented_ds = augmenter.augment(N=0)
    assert len(augmented_ds) == 0

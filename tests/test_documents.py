import pytest

from guided_scdv import Documents


documents = [
    (["これ", "は", "通常", "の", "書類", "です"], 6),
    ([], 0)
]

@pytest.mark.parametrize("test_document,length", documents)
def test_eval(test_document, length):
    assert len(Documents(test_document)) == length

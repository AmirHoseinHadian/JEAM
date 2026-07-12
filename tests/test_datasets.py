from jeam.utility.datasets import load_fennell2023, load_kvam2019


def test_bundled_datasets_load():
    assert not load_fennell2023().empty
    assert not load_kvam2019().empty

import pytest

@pytest.fixture
def dad_fixture():
    from src.pydad.dad_load import DadLoad
    return DadLoad

def test_read_sample(dad_fixture, capsys):
    dl = dad_fixture("/scratch/beapen/pydad/src/pydad/resources/", "dad201617c")
    print(dl.sample.head(5))
    assert dl is not None

def test_read_count(dad_fixture, capsys):
    dl = dad_fixture("/scratch/beapen/pydad/src/pydad/resources/", "dad201617c")
    print(dl.count)
    assert dl.count > 100
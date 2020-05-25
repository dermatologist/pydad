import pytest

@pytest.fixture
def dad_fixture():
    from src.pydad.dad_load import DadLoad
    return DadLoad

@pytest.fixture
def dad_read():
    from src.pydad.dad_read import DadRead
    return DadRead

def test_read_diagnosis(dad_fixture, dad_read, capsys):
    dl = dad_fixture("/scratch/beapen/pydad/src/pydad/resources/", "dad201617c")
    dr = dad_read(dl.sample)
    print(dr.has_diagnosis('E66')) # Obesity
    assert dr.count(dr.has_diagnosis('E66')) > 100


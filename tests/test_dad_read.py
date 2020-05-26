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
    dl = dad_fixture("/home/bell/Documents/work/data/", "dad201617c")
    dr = dad_read(dl.sample)
    print(dr.has_diagnosis('E66')) # Obesity
    assert dr.count(dr.has_diagnosis('E66')) > 100

def test_read_treatment(dad_fixture, dad_read, capsys):
    dl = dad_fixture("/home/bell/Documents/work/data/", "dad201617c")
    dr = dad_read(dl.sample)
    print(dr.has_treatment('1NF80')) # Partial gastrectomy for repair of gastric diverticulum
    assert dr.count(dr.has_treatment('1NF80')) > 10

def test_read_comorbidity(dad_fixture, dad_read, capsys):
    dl = dad_fixture("/home/bell/Documents/work/data/", "dad201617c")
    dr = dad_read(dl.sample)
    print(dr.comorbidity('E66')) # Obesity

def test_read_interventions(dad_fixture, dad_read, capsys):
    dl = dad_fixture("/home/bell/Documents/work/data/", "dad201617c")
    dr = dad_read(dl.sample)
    print(dr.interventions('1NF80')) # Partial gastrectomy for repair of gastric diverticulum

def test_read_vector(dad_fixture, dad_read, capsys):
    dl = dad_fixture("/home/bell/Documents/work/data/", "dad201617c")
    dr = dad_read(dl.sample)
    print(dr.vector(dr.has_diagnosis('E66'), include_treatments=True)) # Obesity

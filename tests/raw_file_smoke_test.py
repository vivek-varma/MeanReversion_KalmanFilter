import pathlib
def test_has_raw_files():
    assert any(pathlib.Path("data/raw/databento").glob("ZNZF_*.parquet"))
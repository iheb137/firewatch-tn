import pytest

from ml.spread_tf.synthetic import write_split


@pytest.fixture(scope="session")
def ndws_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ndws")
    for i, (split, n) in enumerate([("train", 96), ("eval", 32), ("test", 32)]):
        write_split(str(d), split, n, seed=i)
    return str(d)

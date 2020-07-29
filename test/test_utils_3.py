import h5py
import kaldiio
import numpy as np
import pytest



from espnet.utils3.training.batchfy import make_batchset
from test3.utils_test import make_dummy_json

@pytest.mark.parametrize("batch_size", [4, 8])
def test_get_batch(batch_size):
    dummy_json = make_dummy_json(128, [40, 100], [50, 800])
    batches = make_batchset(
        dummy_json,
        "input",
        batch_size,
        60)
    print(batches)
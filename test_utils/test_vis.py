import pytest
import os
import pandas as pd
from utils.vis.vis_fts_emo_relation import vis_fts_emo_scatter


@pytest.mark.parametrize("csv_f", ["test_csv/temp_wav_emofts.csv"])
def test_vis_fts_emo_relation(csv_f, out_f):
    vis_fts_emo_scatter(csv_f, out_f)


if __name__ == '__main__':
    test_vis_fts_emo_relation("test_csv/temp_wav_emofts.csv", "test_csv/temp_wav_emofts.png")
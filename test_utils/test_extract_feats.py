#!/usr/bin/env python3

#from utils.dsp import silenceremove

from utils.extract_txtfeats import extract_txtfeats_from_txt
import pytest
import os
import pandas as pd



@pytest.mark.parametrize("id_txt_f,corp_txt_f",
                         [("test_csv/text.csv"), ("test_csv/test_text_tfidf.csv")]
                         [("test_csv/text.csv"), None])
def test_extract_txt_fts(id_txt_f, corp_txt_f):
    outdict = extract_txtfeats_from_txt(id_txt_f, corp_txt_f)
    print(outdict)



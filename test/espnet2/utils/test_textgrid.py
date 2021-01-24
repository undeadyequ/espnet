import pytest
from utils.textgrid import demo_TextGrid, TextGrid

test_textgrid = "/home/rosen/Project/espnet/tools/force_alignment/emotion_alignment_result/emotion/CB-JE-14-37_angry.TextGrid"
@pytest.mark.parametrize(
    "file, first_endtime",
    [
        (test_textgrid, 0.920)
    ],
)
def test_textgrid(file, first_endtime):
    fid = TextGrid.load(file)
    for iter in fid:
        print("sdf")
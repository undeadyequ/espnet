from kaldiio import ReadHelper
from kaldiio import WriteHelper
import numpy


def read_scp(scp_f: str
             ) -> numpy.array:
    scp_d = dict()
    with ReadHelper('scp:{}'.format(scp_f)) as reader:
        for key, numpy_array in reader:
            scp_d[key] = numpy_array
    return scp_d


def write_scp(scp_f: str,
              scp_d: numpy.array) -> None:
    ark_f = scp_f[-3:] + "ark"
    with WriteHelper('ark,scp:{},{}'.format(ark_f, scp_f)) as writer:
        for k, v in scp_d.items():
            writer(k, v)

        for i in range(10):
            writer(str(i), numpy.random.randn(10, 10))
            # The following is equivalent
            # writer[str(i)] = numpy.random.randn(10, 10)


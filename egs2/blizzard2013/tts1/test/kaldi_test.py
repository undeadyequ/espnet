import numpy
from kaldiio import WriteHelper, ReadHelper
import sklearn
from sklearn.linear_model import LinearRegression


with WriteHelper('ark,scp:file.ark,file.scp') as writer:
    for i in range(10):
        writer(str(i), numpy.random.randn(10, 10))
        # The following is equivalent
        # writer[str(i)] = numpy.random.randn(10, 10)



with ReadHelper("scp:file.scp") as reader:
    for key, array in reader:
        print(key, type(array))
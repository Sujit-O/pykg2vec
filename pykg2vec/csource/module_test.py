import file_handler as fh
import numpy as np

id=1
bs=3
te=2

row=bs
col=te*2+4
func=fh.read_data
data = func(id,bs,te,'./test.txt')
data =np.asarray(data)
data= np.reshape(data,[row,col])
print(data)

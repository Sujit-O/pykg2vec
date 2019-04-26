from ctypes import cdll
import ctypes
import os
import numpy as np
import sys
sys.path.append("./")

idx=0
bs=3
te=2

col=te*2+4
row=bs
total_data = row*col
# file=os.getcwd()+'\\read_data.dll'
file='.\\read_data.dll'
# file=os.getcwd()+'/read_data.so'
fh = cdll.LoadLibrary(file)
func=fh.read_data
func.restype = ctypes.POINTER(ctypes.c_float)
data=func(idx,bs,te,b'./test.txt')
val = []
for i in range(total_data):
    print(i, data[i])
    val.append(data[i])
print(val)
val=np.asarray(val)
val.reshape([row, col])
print(val)
# import pdb
# pdb.set_trace()
# np.ctypeslib.as_array((ctypes.c_double * total_data).from_address(ctypes.addressof(data.contents)))
# print(data)



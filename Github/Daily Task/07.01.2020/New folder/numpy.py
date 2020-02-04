# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:22:41 2020

@author: Raja1
"""

import numpy as np


a= np.array([1,3,5,6,4], dtype='int32')
print(a)

b= np.array([[3.5,5.8,7],[6.7,8,9.9]])
print(b)

# dimension 
a.ndim
b.ndim

# shape
a.shape
b.shape

# type
a.dtype
b.dtype

# Size
a.itemsize
b.itemsize

#total size--Total no of elements * itemsize
a.nbytes
b.nbytes

A= np.array([[3,8,6,8,7],[6,3,6,8,9]])
print(A)
A.shape

#Get the spacific element
A[1,3]

#specific  row
A[0,:]
A[1,:]

#SPECIFIC COLOUMN
A[:,3]

## Get some diff index
A[0, 0:5:2]
 
## change the element 1st row 8
A[1,4]=20
print(A)

## 3D Example

x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
x.shape
print(x)

# get specific element
x[0,1,1]

x[:,1,:]

x[:,0,:]


x[:,1,:]
 










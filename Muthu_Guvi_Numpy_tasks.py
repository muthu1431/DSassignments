#!/usr/bin/env python
# coding: utf-8

# # Numpy
# 
# 

# #### 1. Import the numpy package under the name `np` (★☆☆) 
# (**hint**: import … as …)

# In[1]:


pip install numpy


# #### 2. Print the numpy version and the configuration (★☆☆) 
# (**hint**: np.\_\_version\_\_, np.show\_config)

# In[4]:


import numpy as np
print(np.__version__)
print(np.show_config())


# #### 3. Create a null vector of size 10 (★☆☆) 
# (**hint**: np.zeros)

# In[8]:


x = np.zeros(10)
print(x)


# #### 4.  How to find the memory size of any array (★☆☆) 
# (**hint**: size, itemsize)

# In[9]:


print(x.size * x.itemsize)


# #### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆) 
# (**hint**: np.info)

# In[10]:


print(np.info(np.add))


# #### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 
# (**hint**: array\[4\])

# In[12]:


x[4]=1
print(x)


# #### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) 
# (**hint**: np.arange)

# In[14]:


ttf = np.arange(10,50)
print(ttf)


# #### 8.  Reverse a vector (first element becomes last) (★☆☆) 
# (**hint**: array\[::-1\])

# In[15]:


print(ttf[::-1])


# #### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 
# (**hint**: reshape)

# In[18]:


array = np.arange(0,9)
print(array.reshape(3,3))


# #### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 
# (**hint**: np.nonzero)

# In[20]:


arr = [1,2,0,0,4,0]
print(np.nonzero(arr))


# #### 11. Create a 3x3 identity matrix (★☆☆) 
# (**hint**: np.eye)

# In[21]:


x = np.eye(3)
print(x)


# #### 12. Create a 3x3x3 array with random values (★☆☆) 
# (**hint**: np.random.random)

# In[25]:


x = np.random.random((3,3,3))
print(x)


# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 
# (**hint**: min, max)

# In[29]:


x = np.random.random((10,10))
print(x.min(),x.max())


# #### 14. Create a random vector of size 30 and find the mean value (★☆☆) 
# (**hint**: mean)

# In[30]:


arr = np.random.random((30))
print(arr.mean())


# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
# (**hint**: array\[1:-1, 1:-1\])

# In[33]:


arr = np.ones((5,5))
arr[1:-1,1:-1] = 0
print(arr)


# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) 
# (**hint**: np.pad)

# In[34]:


arr = np.ones((5,5))
arr = np.pad(arr, pad_width = 2)
print(arr)


# #### 17. What is the result of the following expression? (★☆☆) 
# (**hint**: NaN = not a number, inf = infinity)

# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# 0.3 == 3 * 0.1
# ```

# In[ ]:


False


# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 
# (**hint**: np.diag)

# In[35]:


arr = np.diag([0,1,2,3,4])
print(arr)


# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 
# (**hint**: array\[::2\])

# In[37]:


arr = np.zeros((8,8))
arr[1::2,::2] = 1
arr[::2,1::2] = 1
print(arr)


# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 
# (**hint**: np.unravel_index)

# In[39]:


print(np.unravel_index(100,(6,7,8)))


# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) 
# (**hint**: np.tile)

# In[40]:


arr = np.array([[0,1],[1,0]])
print(np.tile(arr,(4,4)))


# #### 22. Normalize a 5x5 random matrix (★☆☆) 
# (**hint**: (x - min) / (max - min))

# In[43]:


ar = np.random.random((5,5))
min, max = ar.min(), ar.max()
z = (ar -min)/ (max-min)
print(z)


# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) 
# (**hint**: np.dtype)

# In[1]:


import numpy as np
RGBA = np.dtype([('red',np.uint8),('green',np.uint8),('blue',np.uint8),('alpha',np.uint8)])
color = np.array((1,2,4,3),dtype = RGBA)
print(color['green'])
type(color)


# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 
# (**hint**: np.dot | @)

# In[2]:


arr1 = np.random.random((5,3))
arr2 = np.random.random((3,2))
arr3 = np.dot(arr1,arr2)
print(arr3)


# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) 
# (**hint**: >, <=)

# In[2]:


import numpy as np
arr = np.arange(16)
arr[(arr>3) & (arr<=8)]*=(-1)
print(arr)


# #### 26. What is the output of the following script? (★☆☆) 
# (**hint**: np.sum)

# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

# In[ ]:


10
10


# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

# In[ ]:


# Z<Z>Z in not legal expression


# #### 28. What are the result of the following expressions?

# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

# In[ ]:


nan
0
-9.22337204e+18


# #### 29. How to round away from zero a float array ? (★☆☆) 
# (**hint**: np.uniform, np.copysign, np.ceil, np.abs)

# In[2]:


import numpy as np
arr = np.random.random((5,5))
arr1 = np.round(arr,0)
arr2 = np.round(arr,1)
arr3 = np.round(arr,2)
print("Array with 0 point\n",arr1,"Array with 1 decimal point\n",arr2,"Array with 2 decimal points\n",arr3)


# #### 30. How to find common values between two arrays? (★☆☆) 
# (**hint**: np.intersect1d)

# In[3]:


import numpy as np
x = np.array([0,1,2,3,4])
y = np.array([3,4,5,6,7])
print(np.intersect1d(x,y))


# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) 
# (**hint**: np.seterr, np.errstate)

# In[1]:


import numpy as np
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0
a = np.seterr(**defaults)


# #### 32. Is the following expressions true? (★☆☆) 
# (**hint**: imaginary number)

# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

# In[ ]:


yes


# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) 
# (**hint**: np.datetime64, np.timedelta64)

# In[2]:


import numpy as np
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday, today, tomorrow)


# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) 
# (**hint**: np.arange(dtype=datetime64\['D'\]))

# In[3]:


import numpy as np
dates = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(dates)


# #### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) 
# (**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))

# In[4]:


import numpy as np
A = np.ones(3)*1
B = np.ones(3)*2
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.dot(A,B)


# #### 36. Extract the integer part of a random array using 5 different methods (★★☆) 
# (**hint**: %, np.floor, np.ceil, astype, np.trunc)

# In[5]:


import numpy as np
a = np.random.uniform(0,10,10)
print(a)
print(a%2)
print(np.floor(a))
print(np.ceil(a))
print(a.astype(str))


# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 
# (**hint**: np.arange)

# In[6]:


import numpy as np
arr = np.zeros((5,5))
arr = arr + np.arange(5)
print(arr)


# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) 
# (**hint**: np.fromiter)

# In[7]:


import numpy as np
arr = np.fromiter(np.arange(10),dtype=int,count=-1)
print(arr)


# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 
# (**hint**: np.linspace)

# In[8]:


import numpy as np
arr = np.linspace(0,1,11,endpoint = False)[1:]
print(arr)


# #### 40. Create a random vector of size 10 and sort it (★★☆) 
# (**hint**: sort)

# In[9]:


import numpy as np
arr = np.random.random(10)
arr.sort()
print(arr)


# #### 41. How to sum a small array faster than np.sum? (★★☆) 
# (**hint**: np.add.reduce)

# In[10]:


import numpy as np
import functools as ft
Z = np.arange(5)
res = ft.reduce(np.add,Z)
print(res)


# #### 42. Consider two random array A and B, check if they are equal (★★☆) 
# (**hint**: np.allclose, np.array\_equal)

# In[1]:


import numpy as np
arr1 = np.random.random(10)
arr2 = np.random.random(10)
res1 = np.array_equal(arr1, arr2)
res2 = np.allclose(arr1,arr2)
print(res1)
print(res2)


# #### 43. Make an array immutable (read-only) (★★☆) 
# (**hint**: flags.writeable)

# In[ ]:


arr = np.ones(10)
arr.flags.writeable = False
arr[0] = 8
print(arr)


# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) 
# (**hint**: np.sqrt, np.arctan2)

# In[2]:


import numpy as np
arr= np.random.randint(10,20,(10,2))
#print(arr)
x,y = arr[:,0], arr[:,1]
#print(x,y)
r = np.sqrt(x**2+y**2)
t = np.arctan2(y,x)
poles = np.c_[r,t]
print(poles)


# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 
# (**hint**: argmax)

# In[3]:


import numpy as np
arr = np.random.random(10)
print(arr)
arr[arr.argmax()] = 0
print(arr)


# #### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) 
# (**hint**: np.meshgrid)

# In[4]:


arr = np.zeros((10,10),[("x",float),('y',float)])
arr['x'],arr['y'] = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))
print(arr)


# ####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) 
# (**hint**: np.subtract.outer)

# In[5]:


X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))


# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 
# (**hint**: np.iinfo, np.finfo, eps)

# In[6]:


for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)


# #### 49. How to print all the values of an array? (★★☆) 
# (**hint**: np.set\_printoptions)

# In[7]:


np.set_printoptions(threshold=float("inf"))
Z = np.random.randint(0,10,(5,5))
print(Z)


# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) 
# (**hint**: argmin)

# In[8]:


Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])


# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) 
# (**hint**: dtype)

# In[1]:


import numpy as np
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)


# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) 
# (**hint**: np.atleast\_2d, T, np.sqrt)

# In[2]:


arr = np.random.random((10,2))
X,Y = np.atleast_2d(arr[:,0], arr[:,1])
distance = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(distance)


# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? 
# (**hint**: astype(copy=False))

# In[1]:


import numpy as np
Z = np.arange(10, dtype=np.float32)
Z_1 = Z.astype(np.float32, copy=False)
Z_1[0] = 2
print(Z)


# #### 54. How to read the following file? (★★☆) 
# (**hint**: np.genfromtxt)

# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

# In[2]:


from io import StringIO

# Fake file
s = StringIO('''1, 2, 3, 4, 5
                6,  ,  , 7, 8
                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)


# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) 
# (**hint**: np.ndenumerate, np.ndindex)

# In[3]:


import numpy as np
arr = np.arange(4).reshape(2,2)
for index in np.ndindex(arr.shape):
    print(index, arr[index])


# #### 56. Generate a generic 2D Gaussian-like array (★★☆) 
# (**hint**: np.meshgrid, np.exp)

# In[5]:


a, b = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
#print(a,b)
c = np.sqrt(a*a+b*b)
#print(c)
d = np.exp(-( (c-0)**2 / ( 2.0 * 1.0**2 ) ) )
print(d)


# #### 57. How to randomly place p elements in a 2D array? (★★☆) 
# (**hint**: np.put, np.random.choice)

# In[6]:


arr = np.zeros((4,4))
np.put(arr, np.random.choice(range(4*4), 5, replace=False),1)
print(arr)


# #### 58. Subtract the mean of each row of a matrix (★★☆) 
# (**hint**: mean(axis=,keepdims=))

# In[7]:


arr = np.random.rand(5, 10)
newarr = arr - arr.mean(axis=1, keepdims = True)
print(newarr)


# #### 59. How to sort an array by the nth column? (★★☆) 
# (**hint**: argsort)

# In[8]:


arr = np.random.randint(0,100,(4,4))
print(arr)
print(arr[arr[:,1].argsort()])


# #### 60. How to tell if a given 2D array has null columns? (★★☆) 
# (**hint**: any, ~)

# In[11]:


arr = np.random.randint(0,3,(2,2))
print(arr)
print((~arr.any(axis=0)).any())


# #### 61. Find the nearest value from a given value in an array (★★☆) 
# (**hint**: np.abs, argmin, flat)

# In[12]:


arr = np.random.uniform(0,1,10)
print(arr)
z = 0.5
m = arr.flat[np.abs(arr - z).argmin()]
print(m)


# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) 
# (**hint**: np.nditer)

# In[13]:


A = np.arange(3).reshape(3,1)
#print(A)
B = np.arange(3).reshape(1,3)
#print(B)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])


# #### 63. Create an array class that has a name attribute (★★☆) 
# (**hint**: class method)

# In[14]:


class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)


# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) 
# (**hint**: np.bincount | np.add.at)

# In[15]:


arr1 = np.ones(10)
#print(arr1)
arr2 = np.random.randint(0,len(arr1),20)
#print(arr2)
np.add.at(arr1,arr2, 1)
print(arr1)


# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) 
# (**hint**: np.bincount)

# In[27]:


arr1 = np.arange(10)
print(arr1)
arr2 =  np.arange(10)
print(arr2)
res = np.bincount(arr1,arr2)
print(res)


# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) 
# (**hint**: np.unique)

# In[28]:


w, h = 256, 256
I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
colors = np.unique(I.reshape(-1, 3), axis=0)
count = len(colors)
print(count)


# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) 
# (**hint**: sum(axis=(-2,-1)))

# In[31]:


arr = np.random.randint(0,10,(3,4,3,4))
#print(arr)
sum = arr.sum(axis=(-2,-1))
print(sum)


# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) 
# (**hint**: np.bincount)

# In[32]:


D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)


# #### 69. How to get the diagonal of a dot product? (★★★) 
# (**hint**: np.diag)

# In[36]:


A = np.random.uniform(0,100,(5,5))
#print(A)
B = np.random.uniform(0,100,(5,5))
#print(B)
np.diag(np.dot(A, B))


# #### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 
# (**hint**: array\[::4\])

# In[38]:


arr = np.array([1,2,3,4,5])
res = np.zeros(len(arr) + (len(arr)-1)*3)
res[::4] = arr
print(res)


# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 
# (**hint**: array\[:, :, None\])

# In[39]:


arr1 = np.ones((5,5,3))
arr2 = 2*np.ones((5,5))
print(arr1 * arr2[:,:,None])


# #### 72. How to swap two rows of an array? (★★★) 
# (**hint**: array\[\[\]\] = array\[\[\]\])

# In[41]:


arr1 = np.arange(25).reshape(5,5)
print(arr1)
arr1[[0,1]] = arr1[[1,0]]
print(arr1)


# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) 
# (**hint**: repeat, np.roll, np.sort, view, np.unique)

# In[ ]:


faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)


# #### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) 
# (**hint**: np.repeat)

# In[42]:


C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)


# #### 75. How to compute averages using a sliding window over an array? (★★★) 
# (**hint**: np.cumsum)

# In[43]:


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))


# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) 
# (**hint**: from numpy.lib import stride_tricks)

# In[46]:


from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)


# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) 
# (**hint**: np.logical_not, np.negative)

# In[47]:


Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)


# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

# In[48]:


def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))


# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

# In[49]:


P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))


# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) 
# (**hint**: minimum, maximum)

# In[ ]:





# #### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) 
# (**hint**: stride\_tricks.as\_strided)

# In[50]:


Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)


# #### 82. Compute a matrix rank (★★★) 
# (**hint**: np.linalg.svd) (suggestion: np.linalg.svd)

# In[51]:


Z = np.random.uniform(0,1,(10,10))
rank = np.linalg.matrix_rank(Z)
print(rank)


# #### 83. How to find the most frequent value in an array? 
# (**hint**: np.bincount, argmax)

# In[52]:


Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())


# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 
# (**hint**: stride\_tricks.as\_strided)

# In[53]:


Z = np.random.randint(0,5,(10,10))
print(sliding_window_view(Z, window_shape=(3, 3)))


# #### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) 
# (**hint**: class method)

# In[ ]:





# #### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) 
# (**hint**: np.tensordot)

# In[54]:


p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)


# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) 
# (**hint**: np.add.reduceat)

# In[55]:


Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)


# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

# In[ ]:





# #### 89. How to get the n largest values of an array (★★★) 
# (**hint**: np.argsort | np.argpartition)

# In[56]:


Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
print (Z[np.argsort(Z)[-n:]])


# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) 
# (**hint**: np.indices)

# In[57]:


def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))


# #### 91. How to create a record array from a regular array? (★★★) 
# (**hint**: np.core.records.fromarrays)

# In[58]:


Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)


# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) 
# (**hint**: np.power, \*, np.einsum)

# In[59]:





# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) 
# (**hint**: np.where)

# In[60]:


A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)


# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

# In[ ]:





# #### 95. Convert a vector of ints into a matrix binary representation (★★★) 
# (**hint**: np.unpackbits)

# In[61]:


I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])


# #### 96. Given a two dimensional array, how to extract unique rows? (★★★) 
# (**hint**: np.ascontiguousarray)

# In[62]:


Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = np.unique(Z, axis=0)
print(uZ)


# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) 
# (**hint**: np.einsum)

# In[63]:


A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)
np.einsum('i->', A)      
np.einsum('i,i->i', A, B) 
np.einsum('i,i', A, B)    
np.einsum('i,j->ij', A, B)   


# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? 
# (**hint**: np.cumsum, np.interp)

# In[64]:





# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) 
# (**hint**: np.logical\_and.reduce, np.mod)

# In[65]:


X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])


# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) 
# (**hint**: np.percentile)

# In[66]:


X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)


# In[ ]:





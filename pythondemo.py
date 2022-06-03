#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 06:20:08 2018

@author: nathanwilliams
"""

# Python program files use a .py extension

# Modules are collections of functions that can be loaded into and used in python
# Modules must be imported
import math

x = math.cos(2*math.pi)
print(x)

# You can import some or all of the functions in a module into the namespace
# Be careful that this does not create conflicting function names

from math import cos, pi

x = cos(2*pi)
print(x)

from math import *

x = cos(2*pi)
print(x)

# Symbols in a module that has been imported can be listed using

print(dir(math))

# You can retrieve module documentation using

help(math.log)

help(math)

# Symbol names can contain letters, numbers and some special characters like _.
# They must also start with a letter. Names are case sensitive.
# Certain keywords cannot be used as names.

x1 = 1

# this will throw an error
#1x = 1

# Assignment operator

x = 1.0

# Python is dynamically typed
type(x)

x = 1
type(x)


# Fundamental data types

# integers
x = 1
type(x)

# float
x = 1.0
type(x)

# boolean
b1 = True
b2 = False
type(b1)

# complex numbers
x = 1.0 - 1.0j
type(x)

# Type casting
x = 1.5
print(x, type(x))

x = int(x)
print(x, type(x))

x = 1.0 - 1.0j
z = complex(x)
print(z, type(z))

#x = float(z)

print(z.real)
print(z.imag)
print(abs(z))

# Operators and comparison

# addition
1 + 2

# subtraction
5 - 3

# multiplication
4 * 10

# division
5 / 3

# integer division
5 // 3

# modulus
5 % 3

# power
4 ** 3

# boolean operators
True and False

True or False

not True

3 in [1,2,3]
3 in [4,5,6]
3 not in [1,2,3]
3 not in [4,5,6]

# comparison operators
2 < 3 # less than
2 > 3 # greater than
2 <= 2 # less than or equal to
2 >= 3 # greater than or equal to
3 == 4 # equal to (note that a single = is the assignment operator)
3 != 4 # not equal to

# objects identical?
x = 2
x is 2
x == 2
x is not 2
x != 2

# strings
s = 'Hello world'
type(s)
len(s) # number of characters in string

s2 = s.replace("world", "test")
print(s2)

# indexing/slicing strings
s[0] # indexing starts at zero
s[0:5] # note that the start character is included but not the stop character
s[:5]
s[6:]
s[::1]
s[::2]
s[1::2]
s[1:6:2] # [start:stop:step]

# concatenate strings
s1 = 'Hello'
s2 = ' '
s3 = 'world'
s = s1 + s2 + s3

# lists
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
type(l)
# indexing/slicing lists (same as strings)
l[0] # indexing starts at zero
l[0:5] # note that the start character is included but not the stop character
l[:5]
l[6:]
l[::1]
l[::2]
l[1::2]
l[1:6:2] # [start:stop:step]

# lists with mixed types
l = [1, 'a', 1.0, 1-1j]

# nested lists
nested_list = [1, [2, [3, [4, [5]]]]]
nested_list[1]
nested_list[1][1]
nested_list[1][1][1]
nested_list[1][1][1][1]
nested_list[1][1][1][1][0]

# type casting a string to a list
s2 = list(s)
print(s2)

# sorting lists
s2.sort()
print(s2)

# modifying lists
l = [] # empty list
l.append('A')
l.append('d')
l.append('d')
print(l)

l[1] = 'p'
l[2] = 'p'
print(l)

l[1:3] = ['d', 'd']
print(l)

l.insert(0, "i")
l.insert(1, "n")
l.insert(2, "s")
l.insert(3, "e")
l.insert(4, "r")
l.insert(5, "t")
print(l)

l.remove('A') # remove elements with specific values
print(l)

del l[7] # remove element in specific location
del l[6]
print(l)

# tuples (immutable lists)
point = (1, 2)
print(point, type(point))

point = 1, 2
print(point, type(point))

# unpack tuple
x, y = point
print("x =", x)
print("y =", y)

# point[0] = 3 # tuples are immutable

# dictionaries
params = {"parameter1" : 1.0,
          "parameter2" : 2.0,
          "parameter3" : 3.0}
print(type(params))
print(params)

print("parameter1 = " + str(params["parameter1"]))
print("parameter2 = " + str(params["parameter2"]))
print("parameter3 = " + str(params["parameter3"]))

# modify dictionary
params["parameter1"] = "A"
params["parameter2"] = "B"

# add a new entry
params["parameter4"] = "D"

print("parameter1 = " + str(params["parameter1"]))
print("parameter2 = " + str(params["parameter2"]))
print("parameter3 = " + str(params["parameter3"]))
print("parameter4 = " + str(params["parameter4"]))

# dates and times
import datetime

# create date, time or datetime object
t = datetime.time(12,1,23)
d = datetime.date(2016,5,28)
dt = datetime.datetime(2016,5,28,12,1,23)

# current time
dnow = datetime.date.today()
print(dnow)
dtnow = datetime.datetime.now()
print(dtnow)

# time components
dt.year
dt.month
dt.day
dt.hour
dt.minute
dt.second
dt.weekday

# time deltas
tdelta = datetime.timedelta(days=10)
datetime.timedelta(hours=240)

# dividing time deltas
tdelta/datetime.timedelta(weeks=1)

# adding/subtracting timedeltas to timestamps
dtnew = dt + tdelta
print(dtnew)

# subtracting timestamps
dtnow - dtnew
dtnew - dtnow

# modify datetime
dt.replace(month=7)

# flow control
# conditional statements
statement1 = False
statement2 = False
if statement1:
    print("statement1 is True") # indentation is important!
elif statement2:
    print("statement2 is True")
else:
    print("statement1 and statement2 are False")
    
statement1 = statement2 = True
if statement1:
    if statement2: # nested if statement
        print("both statement1 and statement2 are True")
        
# loops
# for loops
for x in [1,2,3]:
    print(x)

for x in range(4):
    print(x)
    
for x in range(0,4):
    print(x)

for x in range(0,4,2): # range(start,stop+1,step)
    print(x)
    
l = ['wind', 'solar', 'hydro']
for x in l:
    print('I love ' + x + ' power!')

for key, value in params.items(): # iterate over key-value pairs in dictionary
    print(key + " = " + str(value))

for idx, x in enumerate(range(-3,3)): # iterate over list values and index
    print(idx, x)
    
# list comprehensions
l1 = [x**2 for x in range(0,5)]

# while loops
i = 0
while i < 5:
    print(i)
    i = i + 1
print('done')

i = 0
while i < 5:
    print(i)
    i += 1
print('done')    

# defining functions
def func0():
    print('test')
    
func0()

def func1(s): # use docstrings to document functions
    """
    Print a string 's' and tell how many characters it has
    """
    print(s + " has " + str(len(s)) + " characters")

help(func1) # call docstring for func1

func1('test')

def square(x): # use return to return value from function
    """
    Return the square of x.
    """
    return x ** 2

square(4)

def powers(x): # return multiple values with tuples
    """
    Return a few powers of x.
    """
    return x ** 2, x ** 3, x ** 4

powers(4)

x2, x3, x4 = powers(4)
print(x3)

def myfunc(x, p=2, debug=False):
    if debug:
        print("evaluating myfunc for x = " + str(x) + " using exponent p = " + str(p))
    return x**p

myfunc(5)

myfunc(5, debug=True)

myfunc(p=3, debug=True, x=7) # order is unimportant if input variables are named

# lambda functions
f1 = lambda x: x**2
# is equivalent to
def f2(x):
    return x**2

f1(2), f2(2)

# map is a built-in python function
map(lambda x: x**2, range(-3,4))
list(map(lambda x: x**2, range(-3,4)))

# classes - A class is a structure for representing an object and the
# operations that can be performed on the object.
class Point:
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    def __init__(self, x, y):
        """
        Create a new Point at x, y.
        """
        self.x = x
        self.y = y
    def translate(self, dx, dy):
        """
        Translate the point by dx and dy in the x and y direction.
        """
        self.x += dx
        self.y += dy
    def __str__(self):
        return("Point at [%f, %f]" % (self.x, self.y))

p1 = Point(0, 0) # this will invoke the __init__ method in the Point class
print(p1) # this will invoke the __str__ method

p2 = Point(1, 1)
p1.translate(0.25, 1.5)
print(p1)
print(p2)
          
# modules
import mymodule

help(mymodule)

mymodule.my_variable

mymodule.my_function()

my_class = mymodule.MyClass()
my_class.set_variable(10)
my_class.get_variable()

# exceptions (to manage errors)

#raise Exception("description of the error")

def my_function(arguments):
    if not verify(arguments):
        raise Exception('Invalid arguments')
    
    # rest of function code here...

# try and except statements

try:
    print('test')
    # generate an error: the variable test is not defined
    print(test)
except Exception as e:
    print('Caught an exception: ' + str(e))

# Numpy - multidimensional data arrays

# import numpy module
import numpy as np

# from numpy import * (imports all functions from numpy without the need for prefix)

# Creating numpy arrays
v = np.array([1,2,3,4]) # a vector is a list contained in the array function
print(v)

M = np.array([[1,2],[3,4]]) # a matrix consists of nested lists
print(M)

type(v), type(M)

v.shape # returns dimensions of array
M.shape
M.size # returns number of elements in array
np.size(M)
M.dtype # returns data type of array elements

#M[0,0] = 'hello' # returns error, numpy arrays only accept numerical inputs, 
# data type is determined when array is defined

M = np.array([[1,2],[3,4]], dtype=complex) # we can explicitly define the dataype
M

# array generating functions
# arange
x = np.arange(0,10,1) # start, stop, step
x

x = np.arange(-1, 1, 0.1)
x

# linspace
np.linspace(0, 10, 25) # endpoints included, start, stop, number of points

# logspace
np.logspace(0, 10, 10, base=np.e)

# mgrid
x, y = np.mgrid[0:5, 0:5]
x
y

# random numbers drawn from uniform distribution [0,1]
np.random.rand(5,5)

# random numbers drawn from standard normal distribution
np.random.randn(5,5)

# diagonal matrix
np.diag([1,2,3])

# diagonal matrix with offset
np.diag([1,2,3], k=1)

# zeros
np.zeros((3,3))

# ones
np.ones((3,3))

# importing data
# https://github.com/jrjohansson/scientific-python-lectures/blob/master/stockholm_td_adj.dat
data = np.genfromtxt('stockholm_td_adj.dat')
data.shape

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(data[:,0]+data[:,1]/12.0+data[:,2]/365, data[:,5])
ax.axis('tight')
ax.set_title('temperatures in Stockholm')
ax.set_xlabel('year')
ax.set_ylabel('temperature (C)')

# exporting data
M = np.random.rand(3,3)
np.savetxt('random-matrix.csv', M)

# specify number format
np.savetxt('random-matrix.csv', M, fmt='%.5f')

# specify delimiter
np.savetxt('random-matrix.csv', M, fmt='%.5f', delimiter=',')

# saving data in numpy files
np.save('random-matrix.npy', M)
np.load('random-matrix.npy')

# manipulating arrays
# indexing/slicing
v[0] # single element
M[1,1]
M[1,:] # single row
M[:,1] # single column 

# assignment
M[0,0] = 1
M

M[1,:] = 0
M[:,2] = -1
M

# index slicing in one dimension
A = np.array([1, 2, 3, 4, 5])
A[1:3]

A[1:3] = [-2, -3]
A[::]
A[::2]
A[:3]
A[3:]

A = np.array([1, 2, 3, 4, 5])
A[-1]
A[-3:]

A = np.array([[n+m*10 for n in range(5)] for m in range(5)])
A

A[1:4, 1:4]

A[::2, ::2]

# 'fancy indexing'
row_indices = [1, 2, 3]
A[row_indices]

col_indices = [1, 2, -1]
A[row_indices, col_indices]

# index masks
B = np.array([n for n in range(5)])

row_mask = np.array([True, False, True, False, False])
B[row_mask]

# create mask using conditions
x = np.arange(0, 10, 0.5)
mask = (5 < x) * (x < 7.5)
x[mask]

# where
indices = np.where(mask)
indices
x[indices]

# diagonal
np.diag(A)
np.diag(A, -1)

# take
v2 = np.arange(-3, 3)
v2

row_indices = [1, 3, 5]
v2[row_indices]
v2.take(row_indices)
np.take([-3, -2, -1, 0, 1, 2], row_indices)

# choose
which = [1, 0 , 1, 0]
choices = [[-2, -2, -2, -2], [5, 5, 5, 5]]
np.choose(which, choices)

# linear algebra
# scalar array operations
v1 = np.arange(0,5)
v1 * 2
v1 + 2

A * 2, A + 2

# element-wise array-array operations
A * A
v1 * v1

# multipication of arrays with compatible shapes
A.shape, v1.shape
A * v1

# matrix algebra
np.dot(A, A)
np.dot(A, v1)
np.dot(v1, v1)

# cast as matrix and use normal operations
M = np.matrix(A)
v = np.matrix(v1).T # transpose for column vector

M * M
M * v
v.T * v
v + M * v

# transpose
M.T # works on arrays and matrices
np.transpose(M)

# matrix computations
C = np.matrix([[1, 2], [3, 4]])

# matrix inversion
np.linalg.inv(C)

# determinant
np.linalg.det(C)

# basic statistics
np.mean(data[:, 3])
data[:, 3].mean()
data[:, 3].std()
data[:, 3].var()
data[:, 3].min()
data[:, 3].max()

d = np.arange(0, 10)
d.sum()
np.prod(d + 1)
d.cumsum()
np.cumprod(d + 1)

# computations on subsets of arrays
np.unique(data[:, 1]) # months column of data
mask_feb = data[:, 1] == 2
np.mean(data[mask_feb, 3])

# example: monthly mean temperatures
months = np.arange(1, 13)
monthly_mean = [np.mean(data[data[:, 1] == month, 3]) for month in months]

# computations with higher dimensional data
m = np.random.rand(3,3)

m.max() # global max
m.max(axis=0) # max in each column
m.max(axis=1) # max in each row

# reshaping arrays
# reshape
A
n, m = A.shape
B = A.reshape((1, n*m))
B[0, 0:5] = 5
B
A # python stores references!

# flatten to vector
B = A.flatten()
B[0:5] = 10
B
A # flatten creates a copy

# adding new dimensions
v = np.array([1, 2, 3])
np.shape(v)
v[:, np.newaxis] # make column matrix
v[:, np.newaxis].shape

v[np.newaxis, :] # make row matrix
v[np.newaxis, :].shape

# stacking and repeating arrays
# tile and repeat
a = np.array([[1,2], [3,4]])
np.repeat(a, 3)
np.tile(a, 3)

# concatenation
b = np.array([[5, 6]])
np.concatenate((a,b), axis=0)
np.concatenate((a, b.T), axis=1)

# hstack/vstack
np.vstack((a,b))
np.hstack((a,b.T))

# copy
A = np.array([[1, 2], [3, 4]])
B = A
B[0, 0] = 10
A, B # changing B changes A
B = np.copy(A)
B[0, 0] = -5
A, B

# iterating over elements (iterations are slow and should be avoided if possible)

v = np.array([1,2,3,4])
for element in v:
    print(element)

M = np.array([[1, 2],[3, 4]])

for row in M:
    print('row', row)
    
    for element in row:
        print(element)

# using enumerate
for row_idx, row in enumerate(M):
    print('row_idx', row_idx, 'row', row)
    
    for col_idx, element in enumerate(row):
        print('col_idx', col_idx, 'element', element)
        M[row_idx, col_idx] = element **2

M

# vectorizing functions (to avoid looping over elements)
def Theta(x):
    '''
    Scalar implementation of the Heaviside step function
    '''
    if x >= 0:
        return 1
    else:
        return 0

#Theta(np.array([np.arange(-3,4)]))

Theta_vec = np.vectorize(Theta)

Theta_vec(np.array([np.arange(-3,4)]))

def Theta(x):
    '''
    Vector-aware implementation of the Heaviside step function
    '''
    return 1 * (x >= 0)    

Theta(np.array([np.arange(-3,4)]))
Theta(-1.2), Theta(2.6)

# using arrays in conditions
M

if (M > 5).any():
    print('at least one element in M is larger than 5')
else:
    print('no element in M is larger than 5')
    
if (M > 5).all():
    print('all elements in M are larger than 5')
else:
    print('all elements in M are not larger than 5')
    
# type casting
M.dtype

M2 = M.astype(float)
M2
M2.dtype

M3 = M.astype(bool)
M3

# plotting in python
import matplotlib.pyplot as plt

x = np.linspace(0,5,100)
y = x**2

# create figure object
fig = plt.figure()
# create axes
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

# create plot on axis
axes.plot(x, y, 'r')

# create axis labels
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')


fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes
# main figure
axes1.plot(x, y, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')
# insert
axes2.plot(y, x, 'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('insert title')

# using layout manager
fig, axes = plt.subplots()
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')

fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
fig.tight_layout()

# without loop
fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(x, y, 'r')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('title')

axes[1].plot(y, x, 'r')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('title')
fig.tight_layout()

# setting figure size, aspect ratio and dpi
fig = plt.figure(figsize=(8,4), dpi=100)

# setting figure size and aspect ratio with layout manager
fig, axes = plt.subplots(figsize=(12,3))
axes.plot(x, y, 'r')

axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')

# saving figures
fig.savefig("filename.png")
fig.savefig("filename.png", dpi=200)

# creating legends
ax.legend(["curve1", "curve2", "curve3"])

# or
fig, ax = plt.subplots()

ax.plot(x, x**2, label=r"$y = \alpha^2$") # with LaTeX support
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2) # upper left corner
ax.set_xlabel(r'$\alpha$', fontsize=18)
ax.set_ylabel(r'$y$', fontsize=18)
ax.set_title('title')

# customize line and marker styles
fig, ax = plt.subplots(figsize=(12,6))

ax.plot(x, x+1, color="blue", linewidth=0.25)
ax.plot(x, x+2, color="blue", linewidth=0.50)
ax.plot(x, x+3, color="blue", linewidth=1.00)
ax.plot(x, x+4, color="blue", linewidth=2.00)
         
# possible linestype options ‘-‘, ‘--’, ‘-.’, ‘:’, ‘steps’
ax.plot(x, x+5, color="red", lw=2, linestyle='-')
ax.plot(x, x+6, color="red", lw=2, ls='-.')
ax.plot(x, x+7, color="red", lw=2, ls=':')

# custom dash
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...

# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ... ax.plot(x, x+ 9, color="green", lw=2, ls='--', marker='+')
ax.plot(x, x+10, color="green", lw=2, ls='--', marker='o')
ax.plot(x, x+11, color="green", lw=2, ls='--', marker='s')
ax.plot(x, x+12, color="green", lw=2, ls='--', marker='1') # marker size and color
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)

# other types of plots
n = np.array([0,1,2,3,4,5])
xx = np.linspace(-0.75, 1., 100)

fig, axes = plt.subplots(1, 4, figsize=(12,3))

axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
axes[0].set_title("scatter")

axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5);
axes[3].set_title("fill_between")

# histograms
n = np.random.randn(100000)

fig, axes = plt.subplots(1, 2, figsize=(12,4))

axes[0].hist(n)
axes[0].set_title("Default histogram")
axes[0].set_xlim((min(n), max(n)))

axes[1].hist(n, cumulative=True, bins=50)
axes[1].set_title("Cumulative detailed histogram")
axes[1].set_xlim((min(n), max(n)))

# annotation

fig, ax = plt.subplots()

ax.plot(xx, xx**2, xx, xx**3)

ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green")


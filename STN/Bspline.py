import numpy as np

def h_poly_helper(tt0, tt1, tt2, tt3):
  
  A = np.array([
    [1, 0, -3, 2],
    [0, 1, -2, 1],
    [0, 0, 3, -2],
    [0, 0, -1, 1]], dtype=tt1[-1].dtype)

  first = A[0,0] * tt0  + A[0,1] * tt1 + A[0,2] * tt2 + A[0,3] * tt3
  second = A[1,0] * tt0  + A[1,1] * tt1 + A[1,2] * tt2 + A[1,3] * tt3
  third = A[2,0] * tt0  + A[2,1] * tt1 + A[2,2] * tt2 + A[2,3] * tt3
  fourth = A[3,0] * tt0  + A[3,1] * tt1 + A[3,2] * tt2 + A[3,3] * tt3
 
  output = np.asarray([first, second, third, fourth])

  return output

def h_poly(t):
 
  tt0 = 1
  tt1 = tt0 * t # same dimension as t
  tt2 = np.multiply(tt1, t) # same dimension as t
  tt3 = np.multiply(tt2, t)

  return h_poly_helper(tt0, tt1, tt2 , tt3)


def interp_func(x, y):
 
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = np.concatenate([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):

    I = np.searchsorted(x[1:], xs)  # each xs belong to which spline segment
    dx = (x[I+1]-x[I])

    hh = h_poly((xs-x[I])/dx)
    term1 = np.multiply(hh[0],y[I])
    term2 = np.multiply(np.multiply(hh[1], m[I]) ,dx)
    term3 = np.multiply(hh[2],y[I+1])
    term4 = np.multiply(np.multiply(hh[3],m[I+1]),dx)
    return term1 + term2 + term3 + term4
  return f


def bspline(x, y, xs):
  return interp_func(x,y)(xs)


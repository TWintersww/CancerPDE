import numpy as np
import matplotlib.pyplot as plt

class CancerStates2d:
  """
  Class that contains an instance state and hyperparameters for 
  cancer tumor PDE model in 2d spatial (x, y) and 1d temporal (t) 
  dimensions 

  Important Attributes:
  -------
  N : np.array
    3d array [nt][nx][ny] representing tumor density
  F : np.array
    3d array [nt][nx][ny] representing ECM density
  M : np.array
    3d array [nt][nx][ny] reprsenting MDE concentration

  Xdim : float
  Ydim : float
  Tdim : float

  h : float
    spatial step size
  k : float
    time step size

  nx : int
    number steps x
  ny : int
    number steps y
  nt : int
    number steps t

  Important Methods:

  """



  def __init__(self, Xdim: float, Ydim: float, Tdim: float, h: float, k: float):
    self.Xdim = Xdim
    self.Ydim = Ydim
    self.Tdim = Tdim

    self.h = h
    self.k = k
    
    self.nx = int(round(Xdim / h) + 1)
    self.ny = int(round(Ydim / h) + 1)
    self.nt = int(round(Tdim / k) + 1)

    self.N = np.zeros((self.nt, self.nx, self.ny))
    self.F = np.zeros((self.nt, self.nx, self.ny))
    self.M = np.zeros((self.nt, self.nx, self.ny))

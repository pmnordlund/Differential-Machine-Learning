import numpy as np

def derivative_indicator(x,y,eps=1e-03):
  return np.where(np.abs(x-y)< eps,1,0) / (2*eps) # 1*(ST-K) > eps else 0 

def indicator_approx(x,y,eps=1e-03):
  return np.minimum(1,np.maximum(0,(x-y-eps)/(2*eps)))

def best_indicator_approx(x,y,eps=1e-03):
  return np.where(x>y,1,0) - indicator_approx(x,y,eps)  
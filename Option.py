import numpy as np
from scipy.stats import norm

class Option:
  def __init__(self,strike,start,expiry,spot=1):
    self.spot = spot
    self.strike = strike
    self.start = start
    self.expiry = expiry
      
  def option_class_payoff(self,ST):
    return self.option_payoff(ST)

class Call(Option):
  def __init__(self,strike,start,expiry,spot=1):
    Option.__init__(self, strike, start, expiry,spot)
    
  def option_payoff(self, ST):
    return np.where(ST > self.strike, ST-self.strike ,0)
           
    
class Put(Option):
  def __init__(self,strike,start,expiry,spot=1):
    Option.__init__(self, strike, start, expiry,spot)
  
  def option_payoff(self,ST):
    return np.where(ST < self.strike,self.strike - ST,0)
    
class Digital(Option):
  def __init__(self,strike,start,expiry,spot=1):
    Option.__init__(self, strike, start, expiry,spot)
    
  def option_payoff(self,ST):
    return np.where(ST > self.strike, 1,0) # if ST>K: 1 else. 0 


import numpy as np
from scipy.stats import norm
from Option import *
from Model import *
from Helpers import *

class Bachelier(Model):
    # https://arxiv.org/pdf/2104.08686.pdf
  def __init__(self, sigma, interest):
    Model.__init__(self,sigma, interest)

  def spread_initial_price(self, dt,amount, center, scale):  # uses field varialbes from option so cannot go in Model
    normals = np.random.normal(loc=0.0, scale=1.0, size=(amount,1))
    price = center + (scale * self.sigma * np.sqrt(dt) * normals) 
    return price 
  def get_forward_price(self,spots,dt): # much of the Bachelier model is formualted on forward rates
    return spots * np.exp(self.interest * dt)
    
  def model_delta(self, Option,dt,spots): 
    vol = self.sigma * np.sqrt(dt)
    # spots= range of S0's
    if type(Option) == Call:
      return norm.cdf((spots - Option.strike) / vol)
    elif type(Option) == Put:
      return norm.cdf((spots - Option.strike) / vol) - 1 
    elif type(Option) == Digital:
      foward_rates = self.get_forward_price(spots,Option.expiry-Option.start)
      return norm.pdf((foward_rates- Option.strike) / vol) \
                * (np.exp(self.interest * (Option.expiry-Option.start)))/vol
    else:
      raise Exception("Option not supported, must be Call, Put or Digital")
        
  def model_option_price(self, Option,spots):
    vol = self.sigma * np.sqrt(Option.expiry - Option.start)    
    foward_rates = self.get_forward_price(spots=spots,dt=Option.expiry-Option.start)
    discount = np.exp(- self.interest *(Option.expiry - Option.start))
    d = (foward_rates - Option.strike) / vol
    if type(Option) == Call:
      return discount * ((foward_rates - Option.strike) * norm.cdf(d) + vol * norm.pdf(d))
    elif type(Option) == Put:
      return discount * ((Option.strike - foward_rates) * norm.cdf(-d) + vol * norm.pdf(d))
    elif type(Option) == Digital:
      return norm.cdf((foward_rates - Option.strike)/vol) # according to my calc, it should be 1-() but that looks wrong
    else: 
      raise Exception("Option not supported, must be Call, Put or Digital")

  def get_end_model_price(self,spot,dt,normals,sigma=None,anti=False,return_average=False):
      if not anti:
          if sigma is not None:
              # in case a different sigma is used 
              vol = sigma * np.sqrt(dt)
              return spot * np.exp(self.interest * dt) + vol * normals
          else:
              vol = self.sigma * np.sqrt(dt)
              return spot * np.exp(self.interest * dt) + vol * normals
      else:
          if sigma is not None:
              # in case a different sigma is used 
              vol = sigma * np.sqrt(dt)
              S2a = spot * np.exp(self.interest * dt) + vol * normals
              S2b = spot * np.exp(self.interest * dt) - vol * normals
              if return_average:
                  return (S2a + S2b) * 0.5
              return S2a,S2b
          else:
              vol = self.sigma * np.sqrt(dt)
              S2a = spot * np.exp(self.interest * dt) + vol * normals
              S2b = spot * np.exp(self.interest * dt) - vol * normals
              if return_average:
                  return (S2a + S2b) * 0.5
              return S2a,S2b

  def get_model_prices(self,dt, amount, center, scale, normals_ST):
    spot_start =  self.spread_initial_price(dt=dt,\
                                              amount=amount, center=center, scale=scale)
    spot_end = self.get_end_model_price(spot = spot_start,dt = dt,\
                                      normals = normals_ST)
    return spot_start,spot_end

  def compute_model_D_estimate(self,Option,normals,spot_start,spot_end, epsilon=1e-03, 
                                 D_type="Pathwise",anti=False,return_average=False,tau=None):
    # normals should be the same normals used to generate spot_end. spot_start = simulated S0's
      dt = Option.expiry - Option.start
      if tau is not None:
          dt = tau
      vol = self.sigma * np.sqrt(dt)
      discount = np.exp(self.interest*(dt))
      if not anti:
          if D_type == "LRM": 
              forward_rates = self.get_forward_price(spot_start,Option.expiry-Option.start)
              return Option.option_class_payoff * (spot_end - forward_rates) /(vol**2)
          else:
              if type(Option) == Digital: # mixed method 
                  f_prime = derivative_indicator(spot_end,Option.strike, epsilon)
                  h = best_indicator_approx(spot_end,Option.strike,epsilon)
                  forward_rates = self.get_forward_price(spot_start,dt)
                  return discount * \
                          (f_prime + h * ((spot_end - forward_rates)/(vol**2)))
              else:
                  if type(Option) == Call: #Pathwise estimates 
                      return np.where(spot_end >= Option.strike, 1,0) 
                  else:
                      return np.where(spot_end <= Option.strike, -1,0)
      else:
          # only pathwise on call 
          assert len(spot_end)==2,'Needs to have spot1 and spot2 to make pathwise calculations'
          spot1,spot2 = spot_end
          delta1 = np.where(spot1 >= Option.strike, 1,0)
          delta2 = np.where(spot2 >= Option.strike, 1,0) 
          if return_average:
              return (delta1 + delta2) *  0.5
          return delta1,delta2 



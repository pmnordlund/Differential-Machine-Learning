class Model:
  def __init__(self, sigma,interest):
    self.sigma = sigma
    self.interest = interest
    
  def class_delta(self, Option,dt,spots):
    return self.model_delta(Option,dt,spots)

  def class_spread_initial_price(self, dt,amount, center, scale):
    return self.spread_initial_price(dt,amount, center, scale)
    
  def class_option_price(self, Option,spots):
    return self.model_option_price(Option,spots)
        
  def get_end_class_price(self,spot,dt,normals,sigma=None,anti=False,return_average=False):
    return self.get_end_model_price(spot,dt,normals,sigma,anti,return_average)
      
  def get_class_prices(self, dt, amount, center, scale, normals_ST):
    return self.get_model_prices(dt, amount, center, scale, normals_ST)
        
  def compute_class_D_estimate(self,Option,normals,spot_start,spot_end, epsilon=1e-03, 
                                 D_type="Pathwise",anti=False, return_average=False, tau=None):
    return self.compute_model_D_estimate(Option,normals,spot_start,spot_end, epsilon, 
                                             D_type, anti, return_average, tau)

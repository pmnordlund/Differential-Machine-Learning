# Does this change?
import numpy as np
from scipy.stats import norm
from Option import *
from Model import *
import itertools as it
from Helpers import *

class BlackScholes(Model):
    def __init__(self, sigma, interest):
        Model.__init__(self,sigma, interest)


    def crudeMC(self,Option,dt,spot, sigma,nPaths,anti=False,return_average=False):
      # Conventional MC price estimator with a single S0 input
      # Return_average should always be == False
      if return_average==False:
        if anti==True:
            Z = np.random.normal(size=(nPaths,1))
            
            spot_1,spot_2 = self.get_end_model_price(spot,dt,Z,sigma=sigma,anti=anti,return_average=return_average)
            
            payoff_1 = np.exp(-self.interest*dt)*np.where(spot_1>Option.spot,spot_1-Option.strike,0)
            payoff_2 = np.exp(-self.interest*dt)*np.where(spot_2>Option.spot,spot_2-Option.strike,0)
            payoff = (payoff_1+payoff_2)*0.5
            
            delta_1 = np.where(spot_1 > Option.strike, 1,0) * (spot_1 / spot) * np.exp(-self.interest*dt)
            delta_2 = np.where(spot_2 > Option.strike, 1,0) * (spot_2 / spot) * np.exp(-self.interest*dt)
            delta = (delta_1+delta_2)*0.5
      
            MC_price = np.mean(payoff)
            MC_error = np.std(payoff)/np.sqrt(nPaths)
            delta_est = np.mean(delta)
            delta_est_error = np.std(delta)/np.sqrt(nPaths)

        else:
            Z = np.random.normal(size=(nPaths,1))
            spot_end = self.get_end_model_price(spot,dt,Z,sigma=sigma,anti=anti,return_average=return_average)
            payoff = np.exp(-self.interest*dt)*np.where(spot_end>Option.spot,spot_end-Option.strike,0)
            delta = np.where(spot_end > Option.strike, 1,0) * (spot_end / spot) * np.exp(-self.interest*dt)      

            MC_price = np.mean(payoff)
            MC_error = np.std(payoff)/np.sqrt(nPaths)
            delta_est = np.mean(delta)
            delta_est_error = np.std(delta)/np.sqrt(nPaths)

      else: 
        raise Exception("return_average must be False")

      return MC_price, MC_error,delta_est, delta_est_error
        
    def spread_initial_price(self, dt,amount, center, scale): # uses field varialbes from option so cannot go in Model
        normals = np.random.normal(size=(amount,1)) # made in-function, not used for anything than spreading
        price = center + (scale * self.sigma * np.sqrt(dt) * normals) 
        return price 
    
    def model_delta(self,Option,dt,spots): 
        vol = self.sigma * np.sqrt(dt)
        # Spot = range of S0's
        d1 = (np.log(spots / Option.strike) + (self.interest + self.sigma**2 /2) * \
              (dt)) / vol
        if type(Option) == Call:
            return norm.cdf(d1)
        elif type(Option) == Put:
            return norm.cdf(d1) - 1 
        elif type(Option) == Digital:
            d2 = d1 - vol
            discount = np.exp(- self.interest *(Option.expiry - Option.start))
            return (discount) * norm.pdf(d2) / (spots * vol)
        else:
            raise Exception("Option is not supported, must be Call, Put or Digital")
            

    def model_option_price(self, Option,spots):
        vol = self.sigma * np.sqrt(Option.expiry - Option.start)
        d1 = (np.log(spots / Option.strike) + (self.interest + self.sigma**2 /2) * \
              (Option.expiry - Option.start)) / vol
        d2 = d1 - vol
        discount = np.exp(- self.interest *(Option.expiry - Option.start))
        if type(Option) == Call:
            return spots * norm.cdf(d1) - discount * Option.strike * norm.cdf(d2)
        elif type(Option) == Put:
            return norm.cdf(-d2) * Option.strike * discount - norm.cdf(-d1) * spots
        elif type(Option) == Digital:
            return discount * norm.cdf(d2)
        else: 
            raise Exception("Option not supported, must be Call, Put or Digital")
        
    def get_end_model_price(self,spot,dt,normals,sigma=None,anti=False,return_average=False):
    # Takes in a time fraction, e.g. 1/52, ur just tau=T-t
    # Return_average: if anti and (spot1+spot2)/2 should be returns, else (spot1,spot2) 
        if not anti:
            if sigma is None:
                return spot * np.exp((self.interest - self.sigma**2 * 0.5) \
                  * dt + self.sigma * np.sqrt(dt) * normals)
            else:
                return spot * np.exp((self.interest - sigma**2 * 0.5) \
                  * dt + sigma * np.sqrt(dt) * normals)
        else:
            if sigma is None:
                spot1 = spot * np.exp((self.interest - self.sigma**2 * 0.5) \
                  * dt + self.sigma * np.sqrt(dt) * normals) 
                spot2 = spot * np.exp((self.interest - self.sigma**2 * 0.5) \
                  * dt - self.sigma * np.sqrt(dt) * normals) 
                if return_average:
                    return (spot1+spot2)*0.5
                return spot1,spot2 # do average after they are exported. 
            else:
                spot1 = spot * np.exp((self.interest - sigma**2 * 0.5) \
                  * dt + sigma * np.sqrt(dt) * normals)
                spot2 = spot * np.exp((self.interest - sigma**2 * 0.5) \
                  * dt - sigma * np.sqrt(dt) * normals)
                if return_average:
                    return (spot1+spot2)*0.5
                return spot1,spot2 # do average after they are exported 


    def get_model_prices(self,dt, amount, center, scale, normals_ST):
        spot_start =  self.spread_initial_price(dt=dt,\
                                                  amount=amount, center=center, scale=scale)
        spot_end = self.get_end_model_price(spot = spot_start,dt =dt,\
                                          normals = normals_ST)
        return spot_start,spot_end
    
    def compute_model_D_estimate(self,Option,normals,spot_start,spot_end, epsilon=1e-03, 
                                 D_type="Pathwise",anti=False, return_average=False, tau=None):
        # Need to include a tau = T-t as is may be we need different time-to-expiry
        dt = Option.expiry - Option.start
        if tau is not None:
            dt = tau
        vol = self.sigma * np.sqrt(dt)
        discount = np.exp(- self.interest *(dt))
            # Normals should be the same normals used to generate spot_end. spot_start = simulated S0's
        if not anti:
            if D_type == "LRM":
                return discount * Option.option_class_payoff(spot_end) * normals / (spot_start * vol)
            else:
                if type(Option) == Digital: # Use mixed method for digital 
                    return derivative_indicator(spot_end,Option.strike,epsilon) * spot_end / spot_start \
                    + best_indicator_approx(spot_end,Option.strike,epsilon) * normals / (spot_start * vol)
                else:
                    if type(Option) == Call: # Pathwise estimates 
                        return np.where(spot_end > Option.strike, 1,0) * (spot_end / spot_start) * discount
                    else: 
                        return np.where(spot_end < Option.strike, -1,0) * (spot_end / spot_start) * discount
        else:
            # Pathwise antithetic, only delta and only call 
            assert len(spot_end)==2,'Needs to have spot1 and spot2 to make pathwise calculations' 
            spot1,spot2 = spot_end
            delta1 = np.where(spot1 > Option.strike, 1,0) * (spot1 / spot_start) * discount
            delta2 = np.where(spot2 > Option.strike, 1,0) * (spot2 / spot_start) * discount
            if return_average:
                return (delta1 + delta2) *  0.5
            return delta1,delta2 

                 
    def get_training_data(self,dt, amount, normals, option, center, scale, sigma=0.2, spread_method = "normal",
                          differential = False, anti=False, lower=0.0001,upper=2.5):
        # Gathering function to get all training data
        """
        Spread_method: how should S0 be distributed
        Sigma: if we include a two-dim X, then this should be a grid, e.g. sigma = np.linspace(from=0.2,to=1.0, by =0.05)
        
        Upper and lower is used for unfiform grid
        """
        if not np.isscalar(sigma):
            num_sigmas, = sigma.shape
        else:
            num_sigmas = 1
        # Create X
        if spread_method == "normal":
            S0 = self.spread_initial_price(dt=dt,amount=amount //num_sigmas , center=center, scale=scale)
        elif spread_method == "log-normal":
            S0 = self.spread_inital_price_lognormal(dt=dt,amount=amount//num_sigmas, center=center, scale=scale)
        else:
            S0 = np.linspace(lower, upper, amount//num_sigmas)
            S0 = np.reshape(S0, (-1, 1)) # needs to be a 2d matrix (amount,1)
        
        if not np.isscalar(sigma):
            X_train = np.array([i for i in it.product(S0,sigma)], dtype = 'float32') # simular to expand.grid in R
            np.random.shuffle(X_train) # shuffle 
            
            S0 = X_train[:,[0]]
            sigma = X_train[:,[1]]
        else:
            X_train = S0
            
            
        # Create y
        ST = self.get_end_model_price(spot = S0 ,dt = dt ,normals = normals, sigma = sigma)
        payoff = np.exp(-self.interest * (dt)) * option.option_class_payoff(ST)
        y_train = payoff
        # y antithetic
        if anti:
            ST_tilde = self.get_end_model_price(spot = S0 ,dt = dt ,normals = -normals, sigma = sigma)
            payoff_tilde = np.exp(-self.interest * (dt)) * option.option_class_payoff(ST_tilde)
            payoff_AV = (payoff + payoff_tilde) * 0.5
            y_train = payoff_AV
        
        # Create Z. Currently not considered any contorl variates hereof
        if differential:
            if not np.isscalar(sigma): # if dim(X_train) > 1
                # If vega is included
                vega = self.PW_vega(dt=dt,vols = sigma, S_T = ST,  normals = normals, option=option)
                
                delta = self.compute_model_D_estimate(Option=option,normals=normals,spot_start = S0 ,spot_end = ST)
                
                # If antithetic variates is included
                if anti:
                    vega_tilde = self.PW_vega(dt=dt,vols = sigma, S_T = ST_tilde, normals = -normals, option=option)
                    delta_tilde = self.compute_model_D_estimate(Option=option,normals=-normals,spot_start = S0 ,spot_end = ST_tilde)
                    vega = (vega + vega_tilde) *0.5
                    delta = (delta + delta_tilde) *0.5
                    
                Z_train = np.hstack((delta,vega))
            else:
                delta = self.compute_model_D_estimate(Option=option,normals=normals,spot_start = S0 ,spot_end = ST)
                Z_train = delta
                if anti:
                    delta_tilde = self.compute_model_D_estimate(Option=option,normals=normals,spot_start = S0 ,spot_end = ST_tilde)
                    Z_train = (delta + delta_tilde) * 0.5
        else:
            Z_train = None
        return X_train, y_train, Z_train
 
    def spread_inital_price_lognormal(self, dt,amount, center, scale):
        normals = np.random.normal(size=(amount,1)) # Made in-function, not used for anything than spreading
        return center * np.exp(-0.5 * scale*self.sigma*self.sigma*scale*scale * dt + scale*self.sigma * np.sqrt(dt) * normals)
    
    def bs_price(self,S,K,r,sigma,tau):
        vol = sigma * np.sqrt(tau)
        d1 = (np.log(S / K) + (r + sigma**2 /2) * \
              (tau)) / vol
        d2 = d1 - vol
        discount = np.exp(- r *(tau))
        return S * norm.cdf(d1) - discount * K * norm.cdf(d2)


    # BS greeks 
    def bsDelta(self,spot, strike, vol, tau,r):
        vol_T = vol * np.sqrt(tau)  # Vol = sigma
        d1 = (np.log(spot / strike) + (r + vol**2 /2) * \
              (tau)) / vol_T
        return norm.cdf(d1)

    def model_vega(self,spot, strike, vol, tau,r):
        vol_T = vol * np.sqrt(tau)  # Vol = sigma
        d1 = (np.log(spot / strike) + (r + vol**2 /2) * \
              (tau)) / vol_T
        return spot * norm.pdf(d1) * np.sqrt(tau)
    
    def bsTheta(self,spot, strike, vol, tau,r):
        sqrt_tau = np.sqrt(tau)
        vol_T = vol * sqrt_tau  # Vol = sigma
        d1 = (np.log(spot / strike) + (r + vol**2 /2) * \
              (tau)) / vol_T
        d2 = d1-vol_T
        return -(spot * norm.pdf(d1) * vol) / (2 * sqrt_tau) -\
                r * strike * np.exp(-r * tau) * norm.cdf(d2)
    
    def bsRho(self,spot, strike, vol, tau,r):
        sqrt_tau = np.sqrt(tau)
        vol_T = vol * sqrt_tau  #Vol = Sigma
        d1 = (np.log(spot / strike) + (r + vol**2 /2) * \
              (tau)) / vol_T
        d2 = d1-vol_T
        return strike * tau * np.exp(-r * tau) * norm.cdf(d2)
    
    # BS PW greeks: (delta is repeated for PW)
    def PW_delta(self, tau, S_T, K, r,normals, S0):
        disc = np.exp(-r * tau)
        return disc * np.where(S_T > K, 1,0) * S_T/S0

    def PW_vega(self, tau, S_T, sigma, K, r,normals):
        # Should be same normals used to generate S_T
        disc = np.exp(-r * (tau))
        return disc * S_T * (np.sqrt(tau) * normals - sigma * tau) * np.where(S_T > K, 1,0)

    def PW_theta(self, tau, S_T, sigma, K, r,normals):
        # Note: returns -1*theta_PW
        disc = np.exp(-r * (tau))
        part1 = r * np.where(S_T>K,S_T-K,0)
        part2 = np.where(S_T>K,1,0) * S_T * (r - sigma*sigma*0.5 + (sigma*normals)/(2*np.sqrt(tau)))
        return disc * (-part1 + part2)
    
    def PW_rho(self, tau, S_T, K,r):
        disc = np.exp(-r * (tau))
        return  disc * (np.where(S_T>K,1,0) * S_T * tau - tau * np.where(S_T>K,S_T-K,0))

    def test_set_full(self,start,stop,num,K,sigma,tau,r):
        S0 = np.linspace(start,stop,num).reshape((-1, 1))

        price = self.bs_price(S=S0, K=K,r=r, sigma=sigma, tau=tau)
        delta= self.bsDelta(spot=S0, strike=K, vol=sigma, tau=tau,r=r)
        vega = self.model_vega(spot=S0, strike=K, vol=sigma, tau=tau,r=r)
        theta = self.bsTheta(spot=S0, strike=K, vol=sigma, tau=tau,r=r)
        rho = self.bsRho(spot=S0, strike=K, vol=sigma, tau=tau,r=r)
        return S0, price, delta, vega, theta,rho

    def GBM_ST(self, tau, r,normals, S0,sigma):
        # Need one that does not use any of the model vars 
        return S0 * np.exp((r - sigma*sigma *  0.5) \
                  * tau + sigma * np.sqrt(tau) * normals)

    def GBM_ST(self,tau, r,normals, S0,sigma):
            # Need one that does not use any of the model vars 
            return S0 * np.exp((r - sigma*sigma *0.5) \
                      * tau + sigma * np.sqrt(tau) * normals)

    def MC_greeks(self,S0,K,tau,sigma,r,nPaths):
      
      normals = np.random.normal(size=(nPaths,1))

      ST = self.GBM_ST(tau=tau, r=r,normals=normals, S0=S0,sigma=sigma)
      delta = self.PW_delta(tau=tau, S_T=ST, K=K, r=r,normals=normals, S0=S0)
      vega = self.PW_vega(tau=tau, S_T=ST, sigma=sigma, K=K, r=r,normals=normals)
      theta = -self.PW_theta(tau=tau, S_T=ST, sigma=sigma, K=K, r=r,normals=normals)
      rho = self.PW_rho(tau=tau, S_T=ST, K=K,r=r)


      delta_est = np.mean(delta)
      delta_est_error = np.std(delta)/np.sqrt(nPaths)

      vega_est = np.mean(vega)
      vega_est_error = np.std(vega)/np.sqrt(nPaths)

      theta_est = np.mean(theta)
      theta_est_error = np.std(theta)/np.sqrt(nPaths)

      rho_est = np.mean(rho)
      rho_est_error = np.std(rho)/np.sqrt(nPaths)
      
      return delta_est,vega_est,theta_est,rho_est, delta_est_error,vega_est_error,theta_est_error,rho_est_error

    def get_train_all_vars(self,normals, S0_range=1.0, sigma_range=0.2, tau_range=1.0,r_range=0.0, 
                          K=1.0, differential = False, anti=False): 
      # All ranges are default numbers by conveneince 
      # In case some vars are not lists

      # All the bools are only to determine how the set should look, i.e., what combos of (S,sigma,tau,r) should returned
      if np.isscalar(S0_range):
        m1 = 1
        S_bool = False
      else:
        m1 = S0_range.shape[0]
        S_bool = True

      if np.isscalar(sigma_range):
        m2 = 1
        sig_bool = False
      else:
        m2 = sigma_range.shape[0]
        sig_bool = True  

      if np.isscalar(tau_range):
        m3 = 1
        tau_bool = False
      else:
        m3 = tau_range.shape[0]
        tau_bool = True

      if np.isscalar(r_range):
        m4 = 1
        r_bool = False
      else:
        m4 = r_range.shape[0]
        r_bool = True

      n = sum([S_bool, sig_bool, tau_bool, r_bool]) # sums over true
      m_tilde = normals.shape[1]
      
      m = m1*m2*m3*m4

      # Expand to long array of size m = (m_1 * m_2 * m_3 * m_4) * n
      # Currently only in a system that extends, cannot do e.g. tau and r only 
      if S_bool and not (sig_bool or  tau_bool or r_bool):
        # only S0
        X_train = S0_range
      elif S_bool and sig_bool and not (tau_bool or r_bool):
        # S and sigma
        X_train = np.array(np.meshgrid(S0_range,sigma_range)).reshape(n, m).T  
        S0_range = X_train[:,0].reshape(m,1)
        sigma_range = X_train[:,1].reshape(m,1)
      elif S_bool and sig_bool and tau_bool and not r_bool:
        # S,sigma,T
        X_train = np.array(np.meshgrid(S0_range,sigma_range,tau_range)).reshape(n, m).T
        S0_range = X_train[:,0].reshape(m,1)
        sigma_range = X_train[:,1].reshape(m,1)
        tau_range = X_train[:,2].reshape(m,1)
      elif S_bool and sig_bool and tau_bool and r_bool:
        # S,sigma,T,r
        X_train = np.array(np.meshgrid(S0_range,sigma_range,tau_range,r_range)).reshape(n, m).T
        S0_range = X_train[:,0].reshape(m,1) # needs to be a (mx1) and not (m,) shape since normals is (mxm_tilde)
        sigma_range = X_train[:,1].reshape(m,1)
        tau_range = X_train[:,2].reshape(m,1)
        r_range = X_train[:,3].reshape(m,1)
      elif sig_bool and not (tau_bool or r_bool or S_bool):
        # Only sigma
        X_train = sigma_range
      elif sig_bool and tau_bool and not (r_bool or S_bool):
        # Sigma and tau
        X_train = np.array(np.meshgrid(sigma_range,tau_range)).reshape(n, m).T
        sigma_range = X_train[:,0].reshape(m,1)
        tau_range = X_train[:,1].reshape(m,1)
      elif sig_bool and tau_bool and r_bool and not S_bool:
        # Sigma, tau, r
        X_train = np.array(np.meshgrid(sigma_range,tau_range,r_range)).reshape(n, m).T
        sigma_range = X_train[:,0].reshape(m,1)
        tau_range = X_train[:,1].reshape(m,1)
        r_range = X_train[:,2].reshape(m,1)
      elif tau_bool and not (sig_bool or  S_bool or r_bool):
        # Tau only
        X_train = tau_range
      elif tau_bool and r_bool and not (S_bool or sig_bool):
        # Tau, r
        X_train = np.array(np.meshgrid(tau_range,r_range)).reshape(n, m).T
        tau_range = X_train[:,0].reshape(m,1)
        r_range = X_train[:,1].reshape(m,1)
      elif r_bool and not (sig_bool or  tau_bool or S_bool):
        # r only 
        X_train = r_range
      elif r_bool and sig_bool and not (S_bool or tau_bool):
        # sigma  and  r 
        X_train = np.array(np.meshgrid(sigma_range,r_range)).reshape(n, m).T
        sigma_range = X_train[:,0].reshape(m,1)
        r_range = X_train[:,1].reshape(m,1)
      elif r_bool and S_bool and not (sig_bool or tau_bool):
        # S and r 
        X_train = np.array(np.meshgrid(S0_range,r_range)).reshape(n, m).T
        S0_range = X_train[:,0].reshape(m,1)
        r_range = X_train[:,1].reshape(m,1)
      elif S_bool and sig_bool and r_bool and not tau_bool:
        #S,sigma,r
        X_train = np.array(np.meshgrid(S0_range,sigma_range,r_range)).reshape(n, m).T
        S0_range = X_train[:,0].reshape(m,1)
        sigma_range = X_train[:,1].reshape(m,1)
        r_range = X_train[:,2].reshape(m,1)
      elif S_bool and tau_bool and not (r_bool or sig_bool):
        # S and tau
        X_train = np.array(np.meshgrid(S0_range,tau_range)).reshape(n, m).T
        S0_range = X_train[:,0].reshape(m,1)
        tay_range = X_train[:,1].reshape(m,1)


      ST = self.GBM_ST(tau=tau_range, r=r_range,normals=normals, 
                  S0=S0_range,sigma=sigma_range) # Should return mxm_tilde long matrix 
      discount = np.exp(-r_range * tau_range)
      payoff = discount * np.where(ST>K,ST-K,0.0)

      if anti:
        ST_tilde = self.GBM_ST(tau=tau_range, r=r_range,normals=-normals,
                          S0=S0_range,sigma=sigma_range)
        payoff_tilde = discount * np.where(ST_tilde>K,ST_tilde-K,0.0)
        payoff = (payoff + payoff_tilde) * 0.5
      
      y_train = np.mean(payoff,axis=1).reshape(m,1) # Mean across cols 
      if not differential:
        Z_train = None
      else:
        if not np.isscalar(S0_range):
          delta = self.PW_delta(tau=tau_range, S_T=ST, K=K, r=r_range,
                              normals=normals, S0=S0_range)
        if not np.isscalar(sigma_range):
          vega = self.PW_vega(tau=tau_range, S_T=ST, sigma=sigma_range,
                            K=K, r=r_range,normals=normals)
        if not np.isscalar(tau_range):
          theta = self.PW_theta(tau=tau_range, S_T=ST, sigma=sigma_range,
                              K=K, r=r_range,normals=normals)
        if not np.isscalar(r_range):
          rho = self.PW_rho(tau=tau_range, S_T=ST, K=K,r=r_range)
        if anti:
          if not np.isscalar(S0_range):
            delta_tilde = self.PW_delta(tau=tau_range, S_T=ST_tilde, K=K, r=r_range,
                                normals=-normals, S0=S0_range)
            delta  = np.mean((delta + delta_tilde) * 0.5,axis=1).reshape(m,1) # Average across m_tilde
          if not np.isscalar(sigma_range):
            vega_tilde = self.PW_vega(tau=tau_range, S_T=ST_tilde, sigma=sigma_range,
                              K=K, r=r_range,normals=-normals)
            vega   = np.mean((vega + vega_tilde) * 0.5,axis=1).reshape(m,1)
          if not np.isscalar(tau_range):
            theta_tilde = self.PW_theta(tau=tau_range, S_T=ST_tilde, sigma=sigma_range,
                                K=K, r=r_range,normals=-normals)
            theta  = np.mean((theta + theta_tilde) * 0.5,axis=1).reshape(m,1)
          if not np.isscalar(r_range):
            rho_tilde = self.PW_rho(tau=tau_range, S_T=ST_tilde, K=K,r=r_range)
            rho    = np.mean((rho + rho_tilde) * 0.5,axis=1).reshape(m,1)


        
        else:
          if not np.isscalar(S0_range):
            delta = np.mean(delta,axis=1).reshape(m,1) # Average across m_tilde
          if not np.isscalar(sigma_range):
            vega = np.mean(vega,axis=1).reshape(m,1)
          if not np.isscalar(tau_range):
            theta = np.mean(theta,axis=1).reshape(m,1)
          if not np.isscalar(r_range):
            rho = np.mean(rho,axis=1).reshape(m,1)
        

        if S_bool and not (sig_bool or  tau_bool or r_bool):
          # Only S0
          Z_train = delta
        elif S_bool and sig_bool and not (tau_bool or r_bool):
          # S and sigma
          Z_train = np.hstack((delta,vega))
        elif S_bool and sig_bool and tau_bool and not r_bool:
          # S,sigma,T
          Z_train = np.hstack((delta,vega,theta))
        elif S_bool and sig_bool and tau_bool and r_bool:
          #S,sigma,T,r
          Z_train = np.hstack((delta,vega,theta,rho))
        elif sig_bool and not (tau_bool or r_bool or S_bool):
          # only sigma
          Z_train = vega
        elif sig_bool and tau_bool and not (r_bool or S_bool):
          #sigma and tau
          Z_train = np.hstack((vega,theta))
        elif sig_bool and tau_bool and r_bool and not S_bool:
          #sigma,tau,r
          Z_train = np.hstack((vega,theta,rho))
        elif tau_bool and not (sig_bool or  S_bool or r_bool):
          #tau only
          Z_train = theta
        elif tau_bool and r_bool and not (S_bool or sig_bool):
          #tau,r
          Z_train = np.hstack((theta,rho))
        elif r_bool and not (sig_bool or  tau_bool or S_bool):
          #r only 
          Z_train = rho_tilde
        elif r_bool and sig_bool and not (S_bool or tau_bool):
          # sigma and r 
          Z_train = np.hstack((vega,rho)) 
        elif r_bool and S_bool and not (sig_bool or tau_bool):
          # S and r 
          Z_train = np.hstack((delta,rho))
        elif S_bool and sig_bool and r_bool and not tau_bool:
          #S,sigma,r
          Z_train = np.hstack((delta,vega,rho))
        elif S_bool and tau_bool and not (r_bool or sig_bool):
        # S and tau
          Z_train = np.hstack((delta,theta))

      if differential:
        data = np.hstack((X_train, y_train, Z_train))
        np.random.shuffle(data)
        X_train = data[:,0:n]
        y_train = data[:,n].reshape((-1,1)) # needs to be(m,1) not (m,)
        Z_train = data[:,n+1:]
      else:
        data = np.hstack((X_train, y_train, Z_train))
        np.random.shuffle(data)
        X_train = data[:,0:n]
        y_train = data[:,n].reshape((-1,1))
      return X_train, y_train, Z_train

    def train_vars_S0_sigma(self,normals, S0_range,sigma_range,tau=1.0,r=0.0,K=1.0, differential = False, anti=False,realtype='f'):
        # S0_range and sigma_range needs to be already long format
        m,_= S0_range.shape
        ST = self.GBM_ST(tau=tau, r=r,normals=normals, 
                  S0=S0_range,sigma=sigma_range) # should return m x m_tilde long matrix 
        discount = np.exp(-r * tau)
        payoff = discount * np.where(ST>K,ST-K,0.0)        
        if anti:
            ST_tilde = self.GBM_ST(tau=tau, r=r,normals=-normals, # - normals 
                              S0=S0_range,sigma=sigma_range).astype(realtype)
            payoff_tilde = discount * np.where(ST_tilde>K,ST_tilde-K,0.0)
            payoff = (payoff + payoff_tilde) * 0.5
            del payoff_tilde
        y_train = np.mean(payoff,axis=1).reshape(m,1) # mean across cols
        del payoff       
        if not differential:
            del ST
            Z_train = None
            X_train = np.hstack((S0_range,sigma_range))
            data = np.hstack((X_train,y_train))
            del X_train, y_train
            np.random.shuffle(data)
            X_train = data[:,0:2]
            y_train = data[:,2].reshape((-1,1))
            del data
        else:
            X_train = np.hstack((S0_range,sigma_range))
            delta = self.PW_delta(tau=tau, S_T=ST, K=K, r=r,
                              normals=normals, S0=S0_range).astype(realtype)
            vega = self.PW_vega(tau=tau, S_T=ST, sigma=sigma_range,
                            K=K, r=r,normals=normals).astype(realtype)
            del ST
            if anti:
                delta_tilde = self.PW_delta(tau=tau, S_T=ST_tilde, K=K, r=r,
                                    normals=-normals, S0=S0_range).astype(realtype)
                delta  = np.mean((delta + delta_tilde) * 0.5,axis=1).reshape(m,1) # Average across m_tilde
                del delta_tilde,S0_range
                vega_tilde = self.PW_vega(tau=tau, S_T=ST_tilde, sigma=sigma_range,
                                  K=K, r=r,normals=-normals).astype(realtype)
                vega   = np.mean((vega + vega_tilde) * 0.5,axis=1).reshape(m,1)
                del vega_tilde, ST_tilde,sigma_range
                
            else:
                delta  = np.mean(delta,axis=1).reshape(m,1) # Average across m_tilde
                vega = np.mean(vega,axis=1).reshape(m,1)
            Z_train = np.hstack((delta,vega))
            data = np.hstack((X_train,y_train,Z_train))
            del X_train,y_train,Z_train
            np.random.shuffle(data)
            X_train = data[:,0:2]
            y_train = data[:,2].reshape((-1,1))
            Z_train = data[:,3:]
            del data
        return X_train,y_train,Z_train
            

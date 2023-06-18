# functions used to simulate a hedging experiment 

import numpy as np
import numpy.matlib
import PolyReg
import DifferentialML               # Notebook by Antoine Savine (2021)


def compute_true_hedge_error(model,option,sim,M):
    T,t = option.expiry, option.start
    S0 = option.spot
    start_price = model.class_option_price(Option=option,spots = S0)
    a_0 = model.class_delta(Option=option,dt=T-t,spots=S0)
    dt = (T-t)/M

    intitial_price = start_price
    S = np.repeat(S0,sim).reshape((-1,1))
    portfolio_value = np.repeat(intitial_price,sim).reshape((-1,1))
    a = np.repeat(a_0,sim).reshape((-1,1))
    b = portfolio_value - a*S
    Zdt = np.random.normal(loc = 0.0 ,scale=1.0,size = (sim,M+1))
    for t in range(1,M-1):
        S = model.get_end_class_price(spot=S,dt=dt,normals=Zdt[:,t-1])
        portfolio_value = a * S + b 
        tau = T - dt*t
        a = model.class_delta(Option=option,dt=tau,spots=S)
        b = portfolio_value - a * S 
    S = model.get_end_class_price(spot=S,dt=dt,normals=Zdt[:,M-1])
    payoff = option.option_class_payoff(S)
    portfolio_value = a * S + b
    hedge_error = portfolio_value - payoff
    return np.std(hedge_error)/intitial_price

def compute_simulated_hedge_error(model,option,p,w,sim,nRep,M,batch,spread_method="normal",c=None
                                  ,d=None,D_type="Pathwise",epsilon=1e-03, anti=False,simulseed=0):
    
    if c is None:
        c = option.strike
    if d is None:
        d = 1.0
    t,T = option.start, option.expiry
    S0 = option.spot
    simulseed = np.random.seed(simulseed)
    true_option_price = model.class_option_price(Option=option,spots = S0)
    Z = np.random.normal(0,1,size=(sim,1)) 
    S1 = model.class_spread_initial_price(dt=T-t,amount=sim, 
                                        center=c, scale=d) #spread S0 around c. Same one used for all models 
    if spread_method == "uniform":
            S1 = np.linspace(S1.min(), S1.max(), sim) # use from normals min to max for comparison 
            S1 = np.reshape(S1, (-1, 1)) # needs to be a 2d matrix (amount,1)
    

    X = PolyReg.EstimateX(S1, p, sim)
    Y = PolyReg.EstimateY(S1, p, sim)
    theta = np.zeros((p+1,M))
    dt = (T-t)/M

    for t in range(M):
        tau = T-t*dt
        discount = np.exp(model.interest * (tau))
        if anti:
            S2a,S2b = model.get_end_class_price(spot=S1,dt=tau,normals=Z[:,[0]],anti=True) # same as 1:M and (t-1)*dt in R 
            C = discount * (option.option_class_payoff(S2a) + option.option_class_payoff(S2b)) * 0.5
            S2 = (S2a,S2b)
            dydx = model.compute_class_D_estimate(Option=option,normals = Z, 
                                            spot_start=S1,spot_end=S2,
                                            anti=True,return_average=True, tau = tau) #averages and discounts correctly within the function
        else:    
            S2 = model.get_end_class_price(spot=S1,dt=tau,normals=Z[:,[0]]) # same as 1:M and (t-1)*dt in R 
            C = discount * option.option_class_payoff(S2) # works for any option
            dydx = model.compute_class_D_estimate(Option=option,normals=Z[:,[0]], # same as used to generate ST
                                          spot_start=S1,spot_end=S2, tau =tau) # estimate of delta 
        theta[:,[t]] = PolyReg.thetaReg(w, X, Y, C, dydx)   

    X_0 = PolyReg.EstimateX(np.array([[option.spot]],dtype='float32'), p, 1)
    start_price = np.matmul(X_0,theta[:,[0]])
    a_0 = PolyReg.dF_scalar(theta[:,0],option.spot,p)

    intitial_price = true_option_price 
    S = numpy.matlib.repmat(S0,nRep,batch)
    portfolio_value = numpy.matlib.repmat(start_price,nRep,batch) # perfect hedge if hedge_portfolio = option_value 
    a = numpy.matlib.repmat(a_0,nRep,batch)
    b = portfolio_value - a*S
    Zdt = np.random.normal(loc = 0.0 ,scale=1.0,size = (nRep,batch,M+1))
    for t in range(1,M-1):
        S = model.get_end_class_price(spot=S,dt=dt,normals=Zdt[:,:,t-1])
        portfolio_value = a * S + b 
        for batchstep in range(batch):
            for j in range(nRep):
                a[j,batchstep] = PolyReg.dF_scalar(theta[:,t], S[j,batchstep], p)
        b = portfolio_value - a * S 
    S = model.get_end_class_price(spot=S,dt=dt,normals=Zdt[:,:,M-1])
    payoff = option.option_class_payoff(S)
    portfolio_value = a * S + b
    hedge_error = portfolio_value - payoff
    return np.mean(np.std(hedge_error,axis=0))/intitial_price

def hedging_with_nn(model,option, layers, nRep,c,d, batch, M, sim,spread_method="normal",hidden_units=20, epochs=100,batches_per_epoch=16, min_batch_size=256,
                    differential=False, anti=False, weight_seed=None,
                    description="standard training", simulseed=0,
                     learning_rate_schedule=[ (0.0, 1.0e-4), \
                                              (0.5, 1.0e-02), \
                                              (1.0, 1.0e-4)  ]):
    simulseed = np.random.seed(simulseed) # to ensure same normals are used
    t,T = option.start, option.expiry
    S0 = option.spot
    dt = (T-t)/M
    D_type = "Pathwise"
    true_option_price = model.class_option_price(Option=option,spots = S0)
    Z = np.random.normal(0,1,size=(sim,1)) # only need one column 
    S1 = model.class_spread_initial_price(dt=T-t,amount=sim, 
                                        center=c, scale=d) #spread S0 around c. Same one used for all models 
    if spread_method == "uniform":
            S1 = np.linspace(S1.min(), S1.max(), sim) # use from normals min to max for comparison 
            S1 = np.reshape(S1, (-1, 1)) # needs to be a 2d matrix (amount,1)
    
    neural_models = dict.fromkeys(range(M)) # {0:None,1:None,...}



    for t in range(M):
        tau = T-t*dt
        discount = np.exp(model.interest * (tau))
        if anti:
            S2a,S2b = model.get_end_class_price(spot=S1,dt=tau,normals=Z[:,[0]],anti=True) # same as 1:M and (t-1)*dt in R 
            payoff = discount * (option.option_class_payoff(S2a) + option.option_class_payoff(S2b)) * 0.5
            S2 = (S2a,S2b)
            dydx = model.compute_class_D_estimate(Option=option,normals = Z, 
                                            spot_start=S1,spot_end=S2,
                                            anti=True,return_average=True, tau = tau) #averages and discounts correctly within the function
        else:    
            S2 = model.get_end_class_price(spot=S1,dt=tau,normals=Z[:,[0]]) # same as 1:M and (t-1)*dt in R 
            payoff = discount * option.option_class_payoff(S2) # works for any option
            dydx = model.compute_class_D_estimate(Option=option,normals=Z[:,[0]], # same as used to generate ST
                                          spot_start=S1,spot_end=S2, tau =tau) # estimate of delta 
        
        regressor = DifferentialML.Neural_Approximator(S1,payoff,dydx)
        regressor.prepare(m=sim, differential = differential,
                          hidden_units = hidden_units,
                          hidden_layers = layers,
                          weight_seed = weight_seed)
        regressor.train(description=description,
                        reinit=True,
                        epochs = epochs,
                        batches_per_epoch = batches_per_epoch,
                        min_batch_size = min_batch_size,
                        learning_rate_schedule = learning_rate_schedule
                        ) # have not touched lr schedule or callback
        neural_models[t] = regressor


    start_price, a_0 = neural_models[0].predict_values_and_derivs(np.array([[option.spot]],dtype='float32')) 

    intitial_price = true_option_price 
    S = numpy.matlib.repmat(S0,nRep,batch)
    portfolio_value = numpy.matlib.repmat(start_price,nRep,batch) # portfolio values instansiated at the true option value  
    a = numpy.matlib.repmat(a_0,nRep,batch) # derivatives
    b = portfolio_value - a*S
    Zdt = np.random.normal(loc = 0.0 ,scale=1.0,size = (nRep,batch,M+1))
    price = np.zeros(shape=(nRep,batch))
    for t in range(1,M-1):
        S = model.get_end_class_price(spot=S,dt=dt,normals=Zdt[:,:,t-1])
        portfolio_value = a * S + b
        price, a  =  np.array(neural_models[t].predict_values_and_derivs(S.reshape((-1,1)))).reshape(2,nRep,batch)
        b = portfolio_value - a * S 
    S = model.get_end_class_price(spot=S,dt=dt,normals=Zdt[:,:,M-1])
    payoff = option.option_class_payoff(S)
    portfolio_value = a * S + b
    hedge_error = portfolio_value - payoff
    return np.mean(np.std(hedge_error,axis=0))/intitial_price
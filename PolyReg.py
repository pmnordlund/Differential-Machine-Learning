import numpy as np

# Function returns theta as OLS coefficients        
def OLS(X,Y):
    #beta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    beta = np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)),np.matmul(X.transpose(),Y))
    return beta
    
def EstimateY(S0,p,n):
    Y = np.zeros((n,p+1))
    for j in range(n):    
        for i in range(p+1):
            Y[j,i] = i*S0[j]**(i-1)
    return Y

# Function calculates polynomials from S0     
def EstimateX(S0,p,n):
    X = np.zeros((n,p+1))
    for j in range(n):
        for i in range(p+1):
            X[j,i]=S0[j]**(i)
    return X

# Calculate derivative of polynomial
def dF(theta,S0, p,n):
    delta = np.zeros(n)
    for j in range(n):
        delta[j] = 0
        for i in range(p+1):
            delta[j] = delta[j] + theta[i]*i*S0[j]**(i-1)
    return delta

def dF_scalar(theta,S0,p):
    delta = 0
    for i in range(p+1):
            delta = delta + theta[i]*i*S0**(i-1)
    return delta
        

# Calculates regularized theta
def thetaReg(w,X,Y,C,D):
    X1 = np.linalg.inv(w * np.matmul(X.transpose(),X) + (1 - w) * np.matmul(Y.transpose(),Y))
    X2 = (w * np.matmul(X.transpose(),C) + (1-w)* np.matmul(Y.transpose(),D))
    thetaReg = np.matmul(X1,X2)
    return thetaReg



def compute_prices_and_deltas(model,option,p,n,dt=None,center=None,scale=1,
                              w1=0.5,w2=0.001,D_type="Pathwise",epsilon=1e-03):
    if center is None:
        center = option.spot
    if dt is None:
        dt = option.expiry - option.start

        
    # Prices without regularization 
    normals = np.random.normal(loc=0.0,scale=1.0,size=(n,1))
    S0,ST = model.get_class_prices(dt=dt, amount=n, center=center, scale=scale, normals_ST=normals)
    option_payoff = option.option_class_payoff(ST)
    X = EstimateX(S0=S0,p=p,n=n)
    # deltas with regularization 
    Y = EstimateY(S0=S0, p=p, n=n)
    D = model.compute_class_D_estimate(Option=option,normals=normals,spot_start=S0,
                                       spot_end=ST, epsilon=epsilon, D_type=D_type)
    theta = thetaReg(w=1, X=X, Y=Y, C=option_payoff, D=D) 
    thetaHalfHalf = thetaReg(w=w1, X=X, Y=Y, C=option_payoff, D=D)
    thetaDeltaOnly = thetaReg(w=w2, X=X, Y=Y, C=option_payoff, D=D)
    
    poly_price = np.matmul(X,theta)
    truePrice = model.class_option_price(Option=option,spots=S0)
    trueDelta = model.class_delta(Option=option,dt=dt,spots=S0)
    simulatedDeltas = np.reshape(dF(theta=theta,S0=S0,p=p,n=n), (-1, 1))
    simulatedDeltasHalfHalf = np.reshape(dF(theta=thetaHalfHalf,S0=S0,p=p,n=n), (-1, 1))
    simulatedDeltasDeltaOnly = np.reshape(dF(theta=thetaDeltaOnly,S0=S0,p=p,n=n), (-1, 1))

    # Gather all into one array and sort by S0 ascending 
    all_results = np.hstack((S0,poly_price,truePrice,trueDelta,simulatedDeltas,
                           simulatedDeltasHalfHalf,simulatedDeltasDeltaOnly, simulatedDeltasOneOne))
    all_results = all_results[all_results[:,0].argsort(),]
    
    # Compute MSE
    MSE_poly = float(np.nanmean((truePrice-poly_price)**2))
    MSE_priceonly_delta = float(np.nanmean((trueDelta-simulatedDeltas)**2))
    MSE_deltaonly_delta = float(np.nanmean((trueDelta-simulatedDeltasDeltaOnly)**2))
    MSE_halfhalf_delta = float(np.nanmean((trueDelta-simulatedDeltasHalfHalf)**2))



    return {'S0':all_results[:,0],'poly_price': all_results[:,1],
            'true_price':all_results[:,2], 'true_delta':all_results[:,3],
            'price_only_delta':all_results[:,4],'half_half_delta':all_results[:,5],
            'delta_only_delta':all_results[:,6],'delta_one_one':all_results[:,7],
            'S0_unsorted':S0,
            'option_payoff_unsorted':option_payoff,
            'D_estimates_unsorted':D,
            'MSE_poly':MSE_poly,
            'MSE_priceonly_delta':MSE_priceonly_delta,
            'MSE_deltaonly_delta':MSE_deltaonly_delta,
            'MSE_halfhalf_delta':MSE_halfhalf_delta,}



def compute_prices_and_deltas_uniform(model,option,p,n,dt=None,center=None,scale=0.2,
                              w1=0.5,w2=0.001,D_type="Pathwise",epsilon=1e-03):
    if center is None:
        center = option.spot
    if dt is None:
        dt = option.expiry - option.start

    # Uniform bounds
    lower=0.0001
    upper=2.5
    normals = np.random.normal(loc=0.0,scale=1.0,size=(n,1))

    S0 = np.linspace(lower, upper, n)
    S0 = np.reshape(S0, (-1, 1)) # needs to be a 2d matrix (amount,1)
    ST = model.get_end_model_price(spot=S0,dt=dt,normals=normals)

    option_payoff = option.option_class_payoff(ST)
    X = EstimateX(S0=S0,p=p,n=n)
    # Deltas with regularization 
    Y = EstimateY(S0=S0, p=p, n=n)
    D = model.compute_class_D_estimate(Option=option,normals=normals,spot_start=S0,
                                       spot_end=ST, epsilon=epsilon, D_type=D_type)
    theta = thetaReg(w=1, X=X, Y=Y, C=option_payoff, D=D) 
    thetaHalfHalf = thetaReg(w=w1, X=X, Y=Y, C=option_payoff, D=D)
    thetaDeltaOnly = thetaReg(w=w2, X=X, Y=Y, C=option_payoff, D=D)
    poly_price = np.matmul(X,theta)
    truePrice = model.class_option_price(Option=option,spots=S0)
    trueDelta = model.class_delta(Option=option,dt=dt,spots=S0)
    simulatedDeltas = np.reshape(dF(theta=theta,S0=S0,p=p,n=n), (-1, 1))
    simulatedDeltasHalfHalf = np.reshape(dF(theta=thetaHalfHalf,S0=S0,p=p,n=n), (-1, 1))
    simulatedDeltasDeltaOnly = np.reshape(dF(theta=thetaDeltaOnly,S0=S0,p=p,n=n), (-1, 1))

    # Gather all into one array and sort by S0 ascending 
    all_results = np.hstack((S0,poly_price,truePrice,trueDelta,simulatedDeltas,
                           simulatedDeltasHalfHalf,simulatedDeltasDeltaOnly))
    all_results = all_results[all_results[:,0].argsort(),]
    
    # Compute MSE
    MSE_poly = float(np.nanmean((truePrice-poly_price)**2))
    MSE_priceonly_delta = float(np.nanmean((trueDelta-simulatedDeltas)**2))
    MSE_deltaonly_delta = float(np.nanmean((trueDelta-simulatedDeltasDeltaOnly)**2))
    MSE_halfhalf_delta = float(np.nanmean((trueDelta-simulatedDeltasHalfHalf)**2))

    return {'S0':all_results[:,0],'poly_price': all_results[:,1],
            'true_price':all_results[:,2], 'true_delta':all_results[:,3],
            'price_only_delta':all_results[:,4],'half_half_delta':all_results[:,5],
            'delta_only_delta':all_results[:,6],'delta_one_one':all_results[:,7],
            'S0_unsorted':S0,
            'option_payoff_unsorted':option_payoff,
            'D_estimates_unsorted':D,
            'MSE_poly':MSE_poly,
            'MSE_priceonly_delta':MSE_priceonly_delta,
            'MSE_deltaonly_delta':MSE_deltaonly_delta,
            'MSE_halfhalf_delta':MSE_halfhalf_delta,}


    
def compute_prices_and_deltas_anti(model,option,p,n,dt=None,center=None,scale=0.2,
                              w1=0.5,w2=0.001,D_type="Pathwise",epsilon=1e-03):
    if center is None:
        center = option.spot
    if dt is None:
        dt = option.expiry - option.start

        
    # Prices without regularization 
    normals = np.random.normal(loc=0.0,scale=1.0,size=(n,1))
    S0 = model.spread_initial_price(dt=dt,amount=n, center=center, scale=scale)
    ST = model.get_end_model_price(spot=S0,dt=dt,normals=normals,anti=True) # returns ST as tuple 
    ST1,ST2 = ST

    option_payoff1 = option.option_class_payoff(ST1)
    option_payoff2 = option.option_class_payoff(ST2)
    option_payoff = np.exp(-model.interest*dt) * (option_payoff1 + option_payoff2) * 0.5 # anti
    X = EstimateX(S0=S0,p=p,n=n)
    # Deltas with regularization 
    Y = EstimateY(S0=S0, p=p, n=n)
    D = model.compute_class_D_estimate(Option=option,normals=normals,spot_start=S0,
                                       spot_end=ST, epsilon=epsilon, D_type=D_type,anti=True,return_average=True) # anti 

    theta = thetaReg(w=1, X=X, Y=Y, C=option_payoff, D=D) 
    thetaHalfHalf = thetaReg(w=w1, X=X, Y=Y, C=option_payoff, D=D)
    thetaDeltaOnly = thetaReg(w=w2, X=X, Y=Y, C=option_payoff, D=D)
    poly_price = np.matmul(X,theta)
    truePrice = model.class_option_price(Option=option,spots=S0)
    trueDelta = model.class_delta(Option=option,dt=dt,spots=S0)
    simulatedDeltas = np.reshape(dF(theta=theta,S0=S0,p=p,n=n), (-1, 1))
    simulatedDeltasHalfHalf = np.reshape(dF(theta=thetaHalfHalf,S0=S0,p=p,n=n), (-1, 1))
    simulatedDeltasDeltaOnly = np.reshape(dF(theta=thetaDeltaOnly,S0=S0,p=p,n=n), (-1, 1))

    # Gather all into one array and sort by S0 ascending 
    all_results = np.hstack((S0,poly_price,truePrice,trueDelta,simulatedDeltas,
                           simulatedDeltasHalfHalf,simulatedDeltasDeltaOnly))
    all_results = all_results[all_results[:,0].argsort(),]
    
    # Compute MSE
    MSE_poly = float(np.nanmean((truePrice-poly_price)**2))
    MSE_priceonly_delta = float(np.nanmean((trueDelta-simulatedDeltas)**2))
    MSE_deltaonly_delta = float(np.nanmean((trueDelta-simulatedDeltasDeltaOnly)**2))
    MSE_halfhalf_delta = float(np.nanmean((trueDelta-simulatedDeltasHalfHalf)**2))

    return {'S0':all_results[:,0],'poly_price': all_results[:,1],
            'true_price':all_results[:,2], 'true_delta':all_results[:,3],
            'price_only_delta':all_results[:,4],'half_half_delta':all_results[:,5],
            'delta_only_delta':all_results[:,6],'delta_one_one':all_results[:,7],
            'S0_unsorted':S0,
            'option_payoff_unsorted':option_payoff,
            'D_estimates_unsorted':D,
            'MSE_poly':MSE_poly,
            'MSE_priceonly_delta':MSE_priceonly_delta,
            'MSE_deltaonly_delta':MSE_deltaonly_delta,
            'MSE_halfhalf_delta':MSE_halfhalf_delta,}

    
    
    
    
    
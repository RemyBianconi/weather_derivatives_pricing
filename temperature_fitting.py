import os
import pandas as pd
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import norm,normaltest,probplot
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from statistics import pstdev
import pylab 

path = """path_to_folder"""

os.chdir(path)

data = pd.read_csv('temperature_data.csv')

x =np.array(list(data["Temp"].index))
y = np.array(list(data["Temp"]))

###

# TEMPERATURE DATA

plt.plot(x,y)
plt.show()


###
def form(x,A,B,C,phi):
    return A+B*x+C * np.sin(np.radians(x - phi))

param, covariance = optimize.curve_fit(form, x, y,[ 0, 0, 0 ,0])

fit_cosin = form(x, *param)

plt.scatter(x, y, label='data',alpha=0.5)
plt.plot(x, fit_cosin, 'r-')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(45))
plt.show()

###

projection_mean = y-fit_cosin 
plt.plot(projection_mean)
plt.show()


###

plt.hist(projection_mean,bins=500)
plt.show()

###

probplot(projection_mean, dist="norm", plot=pylab)
pylab.show()
###

absc = projection_mean[:-1].reshape((-1, 1))
ordo = projection_mean[1:]
model = LinearRegression(fit_intercept=True)
model.fit(absc, ordo)
b = model.coef_[0]
a = model.intercept_
x_test=np.linspace(min(absc[:-1]),max(absc[:-1]),num=len(absc)).reshape((-1, 1))
y_pred = model.predict(x_test)

se = mean_squared_error(ordo, y_pred)
print("b : ",b)
print("a : ", a)
print("se : ", se)

plt.scatter(absc,ordo,alpha=0.5)
plt.plot(x_test,y_pred,c="red")
plt.show()



###

# OU MODEL

K = -np.log(b)
theta = a/(1-b)
sigma = se*np.sqrt(-2*np.log(b)/(1-b**2))/10

print("K = ", K)
print("theta : ",theta)
print("sigma : ",sigma)

def OU(x,t,K,theta,sigma):
    res=[x]
    for i in range(t):
        ou_t = res[-1]*np.exp(-K)+theta*(1-np.exp(-K))+sigma*np.sqrt((1-np.exp(-2*K))/(2*K))*np.random.normal(0,1)
        res.append(ou_t)
    return np.array(res)

res= OU(projection_mean[0],len(projection_mean)-1,K,theta,sigma)
plt.plot(res,alpha=0.5,c="red")
plt.plot(projection_mean[:-1],alpha=0.5,c="blue")
plt.show()

###

#Autocorrelation 

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]
    
plt.plot(autocorr(res),c="red")
plt.plot(autocorr(projection_mean[:-1]),c="blue")
plt.show()


###

plt.hist(res,bins=500,color="red",alpha=0.3)
plt.hist(projection_mean[:-1],bins=500,color="blue",alpha=0.3)
plt.show()

###

#Prediction

def pred_T(n,x0,t0):
    new_x = np.array([x0+i for i in range(N)])
    fit_cosin_future = form(new_x, *param)
    ou_test= OU(projection_mean[-1],n-1,K,theta,sigma)
    temprerature_prediction = fit_cosin_future + ou_test
    return temprerature_prediction

    
N = 30*6
temprerature_prediction = pred_T(N,x[-1],y[-1])
plt.plot([i for i in range(len(x)+N)],list(y)+list(temprerature_prediction),c="red")
plt.plot(x,y,c="blue")
plt.show()

###

# MC Simulation
referential =y[-1]
def H(n,x0,t0):
    return (referential*n-np.sum(pred_T(n,x0,t0)))

n_sim=1000

# Average diff between our reference point and the our future prediction
sim = np.array([H(N,x[-1],y[-1]) for i in range(n_sim)])

mu_n = np.mean(sim)
sigma_n=pstdev(sim)
print("mu_n : ",mu_n)
print("sigma_n : ",sigma_n)

plt.plot(sim)
plt.show()

###

# PRICING

#risk free rate
r = 0.05/365

def call_hdd(t,strike,r):
    return( np.exp(-r*t)*((strike-mu_n)*(norm.cdf(alpha_n)-norm.cdf(-mu_n/sigma_n))+sigma_n/np.sqrt(2*np.pi)*(np.exp((-alpha_n**2)/2)-np.exp(-((mu_n/sigma_n)**2)/2))) )
    
strike_array = np.linspace(0,1000,num=10000)
prime_aray=[]
for strike in strike_array:
    alpha_n=(strike-mu_n)/sigma_n
    print("alpha_n : ",alpha_n)

    print("N : ", N)
    print("mu_n", mu_n)
    prime_aray.append(call_hdd(N,strike,r))
    print("CALL HDD Strike ", strike," : ",prime_aray[-1])
    
plt.plot(strike_array,prime_aray)
plt.axvline(mu_n,c="red")
plt.show()
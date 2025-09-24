import numpy as np
import scipy.stats as sp
from scipy.optimize import curve_fit



# The decay function (Probability Density Function)
def decayFunc(x, tau):
    return 1/tau*np.exp(-x/tau)
        

def mle(ts): # for an exponential decay function
    N=len(ts)
    res = 0
    for t in ts:
        res += t
    tau = 1/N*res
    err = res/np.sqrt(N)    
    return (tau,err)


# new data from the book for least square fit
xs = [2.5	,7.5	,12.5]
ys = [21	,13	,3]

# least squared fit 
res = curve_fit(decayFunc,xs,ys)


#potential binning?
#def bin(x,width):
#	return width*floor(x/width)+width/2.0

if __name__ == "__main__":


    listOfDecays = [4.99,4.87,2.59,3.04,3.39,6.20,10.61,7.64,3.92,5.33,4.85,2.39,4.16,6.74,3.53,5.86,5.41,26.25,4.40,10.76,7.08,2.86,33.92,3.03,0.98,5.61,4.89,2.26,10.46,6.51,7.36,2.13,6.45,2.29,21.15,4.07,4.34,5.38,7.69,4.93]
 

   
    Aparams = sp.expon.fit(listOfDecays)
    Bparams = sp.expon.fit(listOfDecays,floc=0)
    tau = mle(listOfDecays)
    print(f'expo fit        tau ={Aparams[1]}')
    print(f'expo fit floc0  tau ={Bparams[1]}')
    print(f'expo fit mle    (tau, err) ={tau}')
    print(f'expo fit square {res}')

    #res=sp.stats.fit(sp.stats.expon,listOfDecay)
    ts = np.linspace(0,40,100)# in minutes

    # making the function into a .txt file
    with open('decayFunc.txt', 'w') as file:
        for t in ts:
            file.write(f'{t}\t{decayFunc(t,Aparams[1])}\n')


    # making the function into a .txt file this is when floc=0
    with open('decayFuncFloc.txt', 'w') as file:
        for t in ts:
            file.write(f'{t}\t{decayFunc(t,Bparams[1])}\n')
    
    # making the function into a .txt file this is when floc=0
    with open('decayTau.txt', 'w') as file:
        for t in ts:
            file.write(f'{t}\t{decayFunc(t,tau[0])}\n')

    # making the decay data into a .data file
    with open('decay.data', 'w') as file:
        for decay in listOfDecays:
            file.write(f'{decay}\n')
    
    

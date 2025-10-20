import numpy as np
from iminuit import Minuit
import scipy.stats as ss


def decayNormBinned(t,A,B):
	binWidth = 5.46 # ms
	return A*(1-np.exp(-B*binWidth))*np.exp(-B*t)

def decayNormBinnedBack(t,A,B,back): 
	binWidth = 5.46 # ms
	return A*(1-np.exp(-B*binWidth))*np.exp(-B*t)+back

def chi2Per(func,*params,bins,ys): # (2.90) from the book
	res = 0
	zipped = zip(bins,ys)
	for b,y in zipped:
		vi = func(b,*params)
		res += np.pow(y-vi,2)/vi
	return res

def chi2Ney(func,*params,bins,ys): # (2.90) from the book
	res = 0
	zipped = zip(bins,ys)
	for b,y in zipped:
		if y == 0:
			res += 0
		else:
			vi = func(b,*params)
			res += np.pow(y-vi,2)/y
	return res

if __name__ == "__main__":
	#####	data 	#####
	data = np.loadtxt('ne17.txt', skiprows=5) # skipping the 5 first rows to remove the start up time
	bins = data[:,0]
	obs = data[:,1]
	
	##### 	data	#####
	
	##### 	fitting	#####
	testPer = lambda A, B: chi2Per(decayNormBinned,A,B,bins=bins,ys=obs)	
	testNey = lambda A, B: chi2Ney(decayNormBinned,A,B,bins=bins,ys=obs)	
	
	mPer = Minuit(testPer,A=9.97e3,B=33.83e-3)
	mNey = Minuit(testNey,A=9.97e3,B=33.83e-3)

	mPer.migrad()
	mNey.migrad()

	mPer.hesse()
	mNey.hesse()
	
	print('testing the chi2 Person')
	print(mPer)
	print('testing the chi2 Neyman')
	print(mNey)
	##### 	fitting	#####
	
	#####	fitting w. background	#####
	testPerBack = lambda A, B, back: chi2Per(decayNormBinnedBack,A,B,back,bins=bins,ys=obs)	
	testNeyBack = lambda A, B, back: chi2Ney(decayNormBinnedBack,A,B,back,bins=bins,ys=obs)	
	
	mPerBack = Minuit(testPerBack,A=9.97e3,B=33.83e-3,back=1)
	mNeyBack = Minuit(testNeyBack,A=9.97e3,B=33.83e-3,back=1)

	mPerBack.migrad()
	mNeyBack.migrad()

	mPerBack.hesse()
	mNeyBack.hesse()
	
	print('testing the chi2 Person w. background')
	print(mPerBack)
	print('testing the chi2 Neyman w. background')
	print(mNeyBack)
	#####	fitting w. background	#####

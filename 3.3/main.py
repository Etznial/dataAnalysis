import numpy as np
from iminuit import Minuit

def lin(x,a,b):
	return a*x+b


def chi2Per(func,*params,xs,ys): # Persons chi squared
	res = 0
	xsys = zip(xs,ys)
	for x,y in xsys:
		v = func(x,*params)
		res += np.pow(y-v,2)/v
	return res

def runTest(func,*params,xs,ys):
	listOfAB=[] # list of above and below
	xsys = zip(xs,ys)
	for x,y in xsys:
		v = func(x,*params)
		if v >= y:
			listOfAB.append('+')
		else :
			listOfAB.append('-')
		
	return listOfAB


if __name__ == "__main__":
	data = np.loadtxt('data.txt')
	bins = data[:,0]
	obs1 = data[:,1]
	obs2 = data[:,2]
	obs3 = data[:,3]
	print(chi2Per(lin,1,10,xs = bins,ys = obs1))
	print(runTest(lin,1,10,xs = bins,ys = obs1))
	
	print(chi2Per(lin,1,10,xs = bins,ys = obs2))
	print(runTest(lin,1,10,xs = bins,ys = obs2))
	
	print(chi2Per(lin,1,10,xs = bins,ys = obs3))
	print(runTest(lin,1,10,xs = bins,ys = obs3))
	



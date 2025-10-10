import numpy as np
from iminuit import Minuit
import scipy.stats as ss


def lin(x,a,b):
	return a*x+b


def chi2PExp(exps,ys): # constraint sum(exps) == sum(ys)
	res = 0
	expsys = zip(exps,ys)
	for e,y in expsys:
		res += np.pow(y-e,2)/e
	return res

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

def runTestExp(exp,ys):
	listOfAB=[] # list of above and below
	expys = zip(exp,ys)
	for e,y in expys:
		if e >= y:
			listOfAB.append('+')
		else :
			listOfAB.append('-')
		
	return listOfAB

if __name__ == "__main__":
	data = np.loadtxt('data.txt')
	bins = data[:,0]
	obs1 = data[:,1]
	obs2 = data[:,2]
	exp = data[:,3]

	ndf = 20 # nubmer of degrees of freedom, 

	obs1chi = chi2PExp(exp,obs1)
	print(obs1chi)
	print(ss.chi2.sf(obs1chi,ndf))
	print(runTestExp(exp,obs1))
	
	obs2chi = chi2PExp(exp,obs1)
	print(obs2chi)
	print(ss.chi2.sf(obs2chi,ndf))
	print(runTestExp(exp,obs2))
	



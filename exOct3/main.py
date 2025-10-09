import numpy as np
from iminuit import Minuit

def normExpo(t,a,b):
	return a*(1-np.exp(-2*b))*np.exp(-b*t)


def expo(t,a,b):
	return a*np.exp(-b*t)


def chi2P(func,*params,xs,ys): # chi squared for Poisson distributed data
	res = 0
	xsys = zip(xs,ys)
	for x,y in xsys:
		vi = func(x,*params)
		res += y*np.log(y/vi)+vi-y	
	return res

if __name__ == "__main__":
	
	ts = [ 0, 2, 4,6,8,10,12,14,16,18] # time bins of width 2
	ns = [30,15,11,9,3, 1, 2, 1, 2, 1] # count numbers for bins of width 2
	
	# the reason for the test function is that Minuit only takes a function and the parameters to be minimized
	# will need further eddeting to take datafiles
	test = lambda a, b: chi2P(normExpo, a, b, xs=ts, ys=ns)
	m = Minuit(test,a=30,b=0.2)
	m.migrad()
	m.hesse()
	print(m)

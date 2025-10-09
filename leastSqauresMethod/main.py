import numpy as np
import scipy.stats as sp
from scipy.optimize import curve_fit


### Chi squared test for every fit
def chiSq(func,xs,ys,yserr):
	res = 0
	xsysyserr = zip(xs,ys,yserr)
	for x,y,yerr in xsysyserr:
		res += np.pow((y-func(x))/yerr,2)
	return res
	



### hard coding the least squares method
def linlsf(pxs,xs,ys,yserr):
	# pxs is the variable that the parameter depends on, so this is a list of functions that takes x so like [x**2,x,1] for ax**2+bx+c
	listOfParams = []
	rows = len(xs)
	#print(f'rows:{rows}')
	cols = len(pxs)
	# constructing the design matrix A
	A = np.zeros(shape=(rows, cols), dtype=float)
	
	for r in range(rows):
		for c in range(cols):
			#print(f'r:{r}\tc:{c}')
			A[r,c]=pxs[c](xs[r])
	#print('A')
	#print(A)
	# A should be constructed now
	# construct identity matrix I 
	I = np.identity(rows,dtype=float)
	
	# construct the covariance matrix V
	# First, get the variances (error squared)
	variances = np.pow(yserr, 2)
	# Then, create a diagonal matrix from the variances
	
	V = np.diag(variances)
	#print('V')
	#print(V)

	AT = A.T
	#print('AT')
	#print(AT)
	
	Vinv = np.linalg.inv(V)
	#print('Vinv')
	#print(Vinv)
	
	ATVinv = np.dot(AT,Vinv)
	#print('ATVinv')
	#print(ATVinv)
	
	
	#print('y')
	#print(ys)

	ATVinvy = np.dot(ATVinv,ys)
	#print('ATVinvy')
	#print(ATVinvy)
	
	H = np.dot(ATVinv,A)
	#print('H')
	#print(H)
	
	Hinv = np.linalg.inv(H)
	#print('Hinv')
	#print(Hinv)
	listOfParams = np.dot(Hinv,ATVinvy)
	return listOfParams


def ls(func,xs,ys,yserr):
	res = 0
	for (x,y,yerr) in (xs,ys,yserr):
		res += np.pow((y-func(x))/yerr,2)
	return res		


if __name__ == "__main__":



	xs = 	[1  ,2  ,3  ,4  ,5  ,6]
	ys = 	[1.0,1.3,0.9,1.8,1.2,2.9]
	yserr = [0.1,0.1,0.3,0.1,0.5,0.2]
	# zipped list
	zipped = zip(xs,ys,yserr)

	# putting the data into a .data file
	with open('data.data', 'w') as file:
		for x,y,yerr in zipped:
 			file.write(f'{x}\t{y}\t{yerr}\n')
	
	
	# making the list of parameters and x's
	def f0(x):
		return x**3
	def f1(x):
		return x**2
	def f2(x):
		return x
	def f3(x):
		return 1
	# for y = ax**3+bx**2+cx+d
	pxs=[f0,f1,f2,f3]
	res=linlsf(pxs,xs,ys,yserr)
	def func3(x):
		return res[0]*x**3+res[1]*x**2+res[2]*x+res[3]
	print(f'for y = ax**3+bx**2+cx+d :{res}')
	print(f'Chi-squared:{chiSq(func3,xs,ys,yserr)}')
	
	# for y = ax**2+bx+c
	pxs=[f1,f2,f3]
	res=linlsf(pxs,xs,ys,yserr)
	def func2(x):
		return res[0]*x**2+res[1]*x+res[2]
	print(f'for y = ax**2+bx+c :{res}')
	print(f'Chi-squared:{chiSq(func2,xs,ys,yserr)}')
	
	# for y = ax+b
	pxs=[f2,f3]
	res=linlsf(pxs,xs,ys,yserr)
	def func1(x):
		return res[0]*x+res[1]
	print(f'for y = bx+c:{res}')
	print(f'Chi-squared:{chiSq(func1,xs,ys,yserr)}')
	
	# for y = a
	pxs=[f3]
	res=linlsf(pxs,xs,ys,yserr)
	def func0(x):
		return res[0]
	print(f'for y = c:{res}')
	print(f'Chi-squared:{chiSq(func0,xs,ys,yserr)}')



	



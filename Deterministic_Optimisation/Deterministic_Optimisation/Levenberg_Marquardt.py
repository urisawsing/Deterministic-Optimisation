import numpy as np
import math
from timeit import default_timer as timer

#the function image
def f(x,y): 
    return 100*(y-x**2)**2+(1-x)**2

#The gradient of the function
def gradf(x,y): 
	gx=400*x**3+x*(2-400*y)-2
	gy=200*(y-x**2)
	return np.array([gx,gy])

#hessian of the function
def hessian(x,y):
	fxx=-400*(y-x**2)+800*x**2+2
	fyy=200
	fxy=-400*x
	H=np.array([[fxx,fxy],[fxy,fyy]])
	return H

#lambda times identity function
def landaI(l,dim):
	I=np.zeros((dim,dim))
	for i in range(dim):
		for j in range(dim):
			if i==j:
				I[i,j]=l
	return I

#inverse of a 2x2 matrix
def inverse(A):
	copy=np.zeros((2,2))
	posd=1
	negd=1
	for i in range(0,2):
		for j in range(0,2):
			if i!=j:
				copy[i,j]=-A[i,j]
				negd=negd*A[i,j]
			if i==j:
				copy[i,j]=A[i-1,j-1]
				posd=posd*A[i,j]
	detA=posd-negd
	return 1/(detA)*copy

#product of a matrix times a vector
def prod(A,v):
	result=np.zeros(2)
	for i in range(0,2):
		for j in range(0,2):
			result[i]=result[i]+(A[i,j]*v[j])
	return result

#step function of LM method
def s(x,y,la):
	H=hessian(x,y)
	lambdaI=landaI(la,2)
	grad=gradf(x,y)
	suma=H+lambdaI
	inv=inverse(suma)
	return -prod(inv,grad)


#main function or Levenberg-Marquardt algorithm
def LVMD(xin,yin,e,Lambda):
	#setting the initial conditions
	start = timer()
	finish=False
	maxiter=int(1e4)
	points=np.zeros((maxiter+2,2))
	posx=xin
	posy=yin
	i=0

	while finish==False:
		#saving points in a point array
		points[i,0]=posx
		points[i,1]=posy

		#step
		sx,sy=s(posx,posy,Lambda)
		xnew=posx+sx
		ynew=posy+sy
		
		#extra calculations for the termination condition
		newgrad=gradf(xnew,ynew)
		diff=np.sqrt((newgrad[0])**2+(newgrad[1])**2)
		fnew=f(xnew,ynew)
		fold=f(posx,posy)

		#landa updating
		if fnew<fold:
			Lambda=Lambda/2
		elif fold<fnew:
			Lambda=2*Lambda

		if diff<e or i>maxiter:
			finish=True

		#updating variables
		i=i+1
		posx=xnew
		posy=ynew

	#some performance calculations
	end = timer()
	t=end-start
	print("Algorithm finished with",i,"iterations and a time of ",t,"seconds", "\nthe final point is (",posx,posy,")")
	return i,t,points
	
 

a,ti,pointv=LVMD(-1.5,-1,1e-9,0.001)
#uncomment this last two lines to get the points and a csv with the points of the algorithm
print(pointv)
np.savetxt("pointsLVMD.csv",pointv,delimiter=",")






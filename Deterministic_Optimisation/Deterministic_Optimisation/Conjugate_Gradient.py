import numpy as np
import math
from timeit import default_timer as timer


#Computes the function 
def f(x,y): 
    return 100*(y-x**2)**2+(1-x)**2


#The gradient of the function
def gradf(x,y): 
    gx=2*(200*x**3-200*x*y+x-1)
    gy=200*(y-x**2)
    return np.array([gx,gy])


#Golden search method 
def golden_search(a, d,delta, pos, direction):
    #initializing parameters
    k=1
    continueGS=True
    xstart=pos[0]
    ystart=pos[1]
    d1=direction[0]
    d2=direction[1]
    phi=((-1+math.sqrt(5))/2)
    
    #algorithm of golden search
    while continueGS:

        #ratio of separation
        step=phi*(d-a)

        #images of intermediate points
        f1=f(xstart+(step+a)*d1,ystart+(step+a)*d2)
        f2=f(xstart+(d-step)*d1,ystart+(d-step)*d2)
        
        #updating searching section
        if f2<f1: 
            a=a
            d=(step+a)
    
        else: 
            d=d
            a=(d-step)
        
        #termination condition
        if (abs(d-a))<delta: 
            search_magnitude=(a+d)/2
            continueGS=False     
        else: 
            k+=1

    return search_magnitude


#scalar product of two vectors
def scalarprod(xv,yv):
    prod=0
    for i in range(2):
        prod=prod+xv[i]*yv[i]
    return prod

#step function for the conjugate gradient method
def s(scalar,v,x0):
    vv=v
    xs=np.zeros(len(vv))
    for i in range(len(vv)):
        vv[i]=vv[i]*scalar
        xs[i]=x0[i]+vv[i]
    return xs 

 
#main function or Conjugate Gradient Method
def cgm(xseed,yseed,len_uncert,delta,maxiter=1e6):
    #initializing some  variables
    itera=0
    start = timer()
    points=np.zeros((int(maxiter+2),2))
    Xk=np.array([xseed,yseed])
    Alg_end=True
    sizek=-gradf(Xk[0],Xk[1])

    #main while
    while Alg_end==True:
        itera=itera+1

        #previous computation of gradient
        grad=gradf(Xk[0],Xk[1]) 
        
        #Compute the step size(alpha) using golden search
        alpha=golden_search(-2,2,delta,Xk,sizek)

        #making the step
        sizecopy=np.array([sizek[0],sizek[1]])
        Xk1=s(alpha,sizecopy,Xk)

        #calculating the beta(Fletcher-Reeves)
        newgrad=gradf(Xk1[0],Xk1[1])
        betak=scalarprod(gradf(Xk1[0],Xk1[1]),gradf(Xk1[0],Xk1[1]))/scalarprod(gradf(Xk[0],Xk[1]),gradf(Xk[0],Xk[1]))

        #saving points
        x=float(Xk1[0])
        y=float(Xk1[1])
        points[itera,0]=x
        points[itera,1]=y

        #final orthogonal direction
        sizek=-gradf(Xk1[0],Xk1[1])+betak*sizek
        
        #termination condition
        if np.sqrt((newgrad[0])**2+(newgrad[1])**2)<len_uncert or itera>maxiter: 
            Alg_end=False
            if itera>maxiter:
                print("point reached by maxiter\n")
            if np.sqrt((Xk[0]-Xk1[0])**2+(Xk[1]-Xk1[1])**2)<len_uncert:
                print("point reached by epsilon condition\n")

        #updating position
        Xk=Xk1   

    #some calculations for performance
    end = timer()
    timef=end-start
    print(end - start)
    print("Algorithm finished with",itera,"iterations and a time of ",timef,"seconds", "\nthe final point is (",Xk[0],Xk[1],")")
    return itera,timef,points


a,t,pointsv=cgm(-1.5,-1,1e-9,1e-8)
#uncomment this last two lines to get the points and a csv with the points of the algorithm
print(pointsv)
np.savetxt("pointsCGM.csv",pointsv,delimiter=",")



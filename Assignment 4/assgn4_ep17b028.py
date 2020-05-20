#Question 1
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.linalg import lstsq

def exp_array(x):
    return np.exp(x)

def coscos_array(x):
    return np.cos(np.cos(x))

x_2pi = np.linspace(0,2*np.pi,101)
x_2pi = x_2pi[:-1]

y1_2pi = exp_array(x_2pi)
y2_2pi = coscos_array(x_2pi)

x = np.linspace(-2*np.pi,4*np.pi,301)
x = x[:-1]

y1 = np.zeros(len(x))   #periodic exp(x)
y2 = np.zeros(len(x))   #periodic cos(cos(x))

for i in range(len(x)):
    y1[i] = y1_2pi[i%100]
    y2[i] = y2_2pi[i%100]

#Plot exp in semilogy scale and cos(cos(x)) in linear scale
plt.plot(x,y1)
plt.yscale('log')
plt.ylabel(r'$e^{x}$',size=20)
plt.xlabel('x',size=20)
plt.title(r'$2\pi$'+' Periodic '+r'$e^{x}$',size=20)
plt.grid(True)
plt.savefig('Figure 1.png')
plt.show()

plt.plot(x,y2)
plt.title(r'$2\pi$'+' Periodic '+r'$cos(cos(x))$',size=20)
plt.ylabel(r'$cos(cos(x))$',size=20)
plt.xlabel('x',size=20)
plt.grid(True)
plt.savefig('Figure 2.png')
plt.show()

#Question 2

def u(x,f,k):
    y = f(x)*np.cos(k*x)
    return y

def v(x,f,k):
    y = f(x)*np.sin(k*x)
    return y

#This function takes in as input a function, number of desired coefficients 
#and returns the array of fourier coefficients of sin, cosine and both 
#appended in the format asked

def coeff(f,Ncoeff):
    coeffarray = np.zeros(2*Ncoeff+1)   #complete array of coefficients
    for k in range(1,len(coeffarray)):
        if(k%2==1):
            coeffarray[k] = (1/np.pi)*scipy.integrate.quad(u,0,2*np.pi,args=(f,k//2+1))[0]
        else:
            coeffarray[k] = (1/np.pi)*scipy.integrate.quad(v,0,2*np.pi,args=(f,k//2))[0]
            coeffarray[0] = (1/(2*np.pi))*scipy.integrate.quad(f,0,2*np.pi)[0]

    return coeffarray
#Function for parsing the complete coefficient array to yield cosine and sine
#array separately
def coeffparse(w):
    a = np.zeros(int((len(w)+1)/2))
    b = np.zeros(len(a)-1)
    a[0] = w[0]
    for i in range(1,len(w)):
        if(i%2==1):
            a[i//2+1] = w[i]
        else:
            b[i//2-1] = w[i]
    return a,b

expcoeff = coeff(exp_array,25)
aexp,bexp = coeffparse(expcoeff)

coscoscoeff = coeff(coscos_array,25)
acoscos,bcoscos = coeffparse(coscoscoeff)
#Question 3
x2 = np.arange(1,26,1)
x1 = np.arange(0,26,1)

#Coefficients in semilog scale for exp(x)
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(x1,np.abs(aexp),'ro')
ax[0].plot(x1,np.abs(aexp),'r--')
ax[0].set_title('Fourier coefficients(cos) in semilogy for'+r'$e^{x}$',size=20)
ax[0].set_ylabel(r'$a_{n}$',size=20)
ax[0].set_xlabel('n',size=20)
ax[0].set_yscale('log')

ax[1].plot(x2,np.abs(bexp),'ro')
ax[1].plot(x2,np.abs(bexp),'r--')
ax[1].set_title('Fourier coefficients(sin) in semilogy for'+r'$e^{x}$',size=20)
ax[1].set_ylabel(r'$b_{n}$',size=20)
ax[1].set_xlabel('n',size=20)
ax[1].set_yscale('log')
plt.savefig('Figure 3.png')
plt.show()

#Coefficients in loglog scale for exp(x)
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(x1,np.abs(aexp),'ro')
ax[0].plot(x1,np.abs(aexp),'r--')
ax[0].set_title('Fourier coefficients(cos) in loglog for'+r'$e^{x}$',size=20)
ax[0].set_ylabel(r'$a_{n}$',size=20)
ax[0].set_xlabel('n',size=20)
ax[0].set_yscale('log')
ax[0].set_xscale('log')

ax[1].plot(x2,np.abs(bexp),'ro')
ax[1].plot(x2,np.abs(bexp),'r--')
ax[1].set_title('Fourier coefficients(sin) in loglog for'+r'$e^{x}$',size=20)
ax[1].set_ylabel(r'$b_{n}$',size=20)
ax[1].set_xlabel('n',size=20)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
plt.savefig('Figure 4.png')
plt.show()

#Coefficients in semilog scale for cos(cos(x))
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(x1,np.abs(acoscos),'ro')
ax[0].plot(x1,np.abs(acoscos),'r--')
ax[0].set_title('Fourier coefficients(cos) in semilogy for cos(cos(x))',size=15)
ax[0].set_ylabel(r'$a_{n}$',size=20)
ax[0].set_xlabel('n',size=20)
ax[0].set_yscale('log')

ax[1].plot(x2,np.abs(bcoscos),'ro')
ax[1].plot(x2,np.abs(bcoscos),'r--')
ax[1].set_title('Fourier coefficients(sin) in semilogy for cos(cos(x))',size=15)
ax[1].set_ylabel(r'$b_{n}$',size=20)
ax[1].set_xlabel('n',size=20)
ax[1].set_yscale('log')
plt.savefig('Figure 5.png')
plt.show()

#Coefficients in loglog scale for cos(cos(x))
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(x1,np.abs(acoscos),'ro')
ax[0].plot(x1,np.abs(acoscos),'r--')
ax[0].set_title('Fourier coefficients(cos) in loglog for cos(cos(x))',size=15)
ax[0].set_ylabel(r'$a_{n}$',size=20)
ax[0].set_xlabel('n',size=20)
ax[0].set_yscale('log')
ax[0].set_xscale('log')

ax[1].plot(x2,np.abs(bcoscos),'ro')
ax[1].plot(x2,np.abs(bcoscos),'r--')
ax[1].set_title('Fourier coefficients(sin) in loglog for cos(cos(x))',size=15)
ax[1].set_ylabel(r'$b_{n}$',size=20)
ax[1].set_xlabel('n',size=20)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
plt.savefig('Figure 6.png')
plt.show()

#Question 4 and 5
def fourierlstsq(x,f,Ncoeff):
    A = np.zeros((len(x),2*Ncoeff+1))
    b = f(x)
    A[:,0] = 1
    for cols in range(1,Ncoeff+1):
        A[:,2*cols-1] = np.cos(cols*x)
        A[:,2*cols] = np.sin(cols*x)
    c = lstsq(A,b)[0]
    return c, A

x = np.linspace(0,2*np.pi,401)
x = x[:-1]

c_exp, A_exp = fourierlstsq(x,exp_array,25)
a_cexp,b_cexp = coeffparse(c_exp)

c_coscos, A_coscos = fourierlstsq(x,coscos_array,25)
a_ccoscos,b_ccoscos = coeffparse(c_coscos)

#Coefficients in loglog scale for exp(x): Integration vs lstsq
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(x1,np.abs(aexp),'ro',label='Coefficients by Integration')
ax[0].plot(x1,np.abs(a_cexp),'go',label='Coefficients by least squares')
ax[0].set_title('Fourier coefficients(cos) in loglog for'+r'$e^{x}$',size=20)
ax[0].set_ylabel(r'$a_{n}$',size=20)
ax[0].set_xlabel('n',size=20)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].legend()

ax[1].plot(x2,np.abs(bexp),'ro',label='Coefficients by Integration')
ax[1].plot(x2,np.abs(b_cexp),'go',label='Coefficients by least squares')
ax[1].set_title('Fourier coefficients(sin) in loglog for'+r'$e^{x}$',size=20)
ax[1].set_ylabel(r'$b_{n}$',size=20)
ax[1].set_xlabel('n',size=20)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].legend()
plt.savefig('Figure 7.png')
plt.show()

#Coefficients in semilogy scale for cos(cos(x)): Integration vs lstsq
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(x1,np.abs(acoscos),'ro',label='Coefficients by Integration')
ax[0].plot(x1,np.abs(a_ccoscos),'go',label='Coefficients by least squares')
ax[0].set_title('Fourier coefficients(cos) in loglog for cos(cos(x))',size=15)
ax[0].set_ylabel(r'$a_{n}$',size=20)
ax[0].set_xlabel('n',size=20)
ax[0].set_yscale('log')
ax[0].legend()

ax[1].plot(x2,np.abs(bcoscos),'ro',label='Coefficients by Integration')
ax[1].plot(x2,np.abs(b_ccoscos),'go',label='Coefficients by least squares')
ax[1].set_title('Fourier coefficients(sin) in loglog for cos(cos(x))',size=15)
ax[1].set_ylabel(r'$b_{n}$',size=20)
ax[1].set_xlabel('n',size=20)
ax[1].set_yscale('log')
ax[1].legend()
plt.savefig('Figure 8.png')
plt.show()

#Question 6
expdiff = np.abs(expcoeff-c_exp)
coscosdiff = np.abs(coscoscoeff-c_coscos)
maxdevexp = max(expdiff)
maxdevcoscos = max(coscosdiff)
print('Maximum deviation in exponential coefficients array is ',maxdevexp)
print('Maximum deviation in cos(cos(x)) coefficients array is ',maxdevcoscos)

#Question 7
plt.figure(figsize=(10,10))
plt.plot(x,exp_array(x),label='True function')
plt.plot(x,np.dot(A_exp,c_exp),label='Fourier reconstruction(lstsq)')
plt.plot(x,np.dot(A_exp,expcoeff),label='Fourier reconstruction(integration)')
plt.title(r'$e^{x}$'+' Actual value vs lstsq',size=20)
plt.ylabel(r'$e^{x}$',size=20)
plt.xlabel('x',size=20)
plt.legend()
plt.savefig('Figure 9.png')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(x,coscos_array(x),label='True function')
plt.plot(x,np.dot(A_coscos,c_coscos),label='Fourier reconstruction(lstsq)')
plt.plot(x,np.dot(A_coscos,coscoscoeff),label='Fourier reconstruction(integration)')
plt.title('cos(cos(x))'+' Actual value vs lstsq',size=20)
plt.ylabel('cos(cos(x))',size=20)
plt.xlabel('x',size=20)
plt.legend()
plt.savefig('Figure 10.png')
plt.show()
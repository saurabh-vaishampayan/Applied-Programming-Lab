import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

#Load the dat file into dat_array
dat_array = np.loadtxt('fitting.dat')

arraystdev = np.logspace(-1,-3,9)
list_labels = [None]*9

for i in range(9):
    list_labels[i] = str(arraystdev[i])

t = np.zeros(101)
for i in range(101):
    t[i] = dat_array[i][0]

#Create column vectors for data of each of the 9 instances
data_matrix = np.zeros((9,101))
for k in range(1,10):
    for i in range(101):
        data_matrix[k-1][i] = dat_array[i][k]
#Plot the raw data with labels
fig1, ax1 = plt.subplots(figsize=(10,6))
for i in range(9):
    ax1.plot(t,data_matrix[i],label=r'$\sigma = $'+list_labels[i])

ax1.set_xlabel(r'$t$',size=20)
ax1.set_ylabel(r'$f(t)+n$',size=20)
ax1.set_title(r'Figure 0')
ax1.legend()
ax1.grid(True)

#Function definition of g
def g(x, a, b):
    y=a*sp.jn(2,t)+b*t
    return y
#Definition of true values
A_true = 1.05
B_true = -0.105

#Add the pure signal to the earlier plot
trueval = g(t,A_true,B_true)
ax1.plot()
plt.plot(t,trueval,'k-',label='True Value')
plt.xlabel(r'$t$',size=20)
plt.ylabel(r'$f(t)+n$',size=20)
plt.title(r'Figure 0')
plt.legend()
plt.grid(True)
plt.savefig('Plot 1.png')
plt.show()

#Plot data of first column with errorbars along with true function
plt.figure(figsize=(10,6))
plt.plot(t,trueval,'k-',label='True value')
plt.errorbar(t[::5], data_matrix[0][::5],0.1,fmt='ro',label='Errorbar for '+r'$\sigma = $'+'0.1')
plt.legend()
plt.xlabel(r'$t$',size=20)
plt.ylabel('y',size=20)
plt.title('True function and errorbar',size=20)
plt.savefig('Errorbar.png')
plt.show()

#Create matrix M in the format asked. This is later used for least squares fitting
jn = sp.jn(2,t)
M = np.c_[jn,t]
p = [A_true,B_true]
g_0 = np.dot(M,p)

#Meshgrid of A and B for contour plots
A = np.arange(0,2,0.1)
B = np.arange(-0.20,0,0.01)
err = np.zeros((2,len(A),len(B)))
a, b = np.meshgrid(A,B)

#Calculate and contour plot error values for A,B belonging to the mesh
for cols in range(2):
        for i in range(len(A)):
            for k in range(len(B)):
                dummy = g(t,A[i],B[k])
                for w in range(len(t)):
                    err[cols][i][k] += (data_matrix[cols][w]-dummy[w])**2
                err[cols][i][k] = (1/101)*err[cols][i][k]
        
        plt.figure(figsize=(10,6))
        cp = plt.contour(a,b,err[cols])
        plt.clabel(cp,inline=True,fontsize=10)
        plt.plot(A_true,B_true,'ro')
        plt.annotate('True Value',(A_true,B_true))
        plt.title('Contour plot for %d st column' %(cols+1))
        plt.xlabel('A',size=20)
        plt.ylabel('B',size=20)
        plt.savefig('Contourplots%d' %(cols+1))
        plt.show()

#Calculate a estimates and errors in a and b by lstsq
a_estimate = np.zeros(9)
b_estimate = np.zeros(9)
a_err = np.zeros(9)
b_err = np.zeros(9)

for i in range(9):
    a_estimate[i], b_estimate[i] = lstsq(M,data_matrix[i])[0]
    a_err[i] = np.abs(A_true-a_estimate[i])**2
    b_err[i] = np.abs(B_true-b_estimate[i])**2

#Linear scale plot of errors in estimate vs stdev
plt.figure(figsize=(10,6))
plt.plot(arraystdev,a_err,'ro',label='A error')
plt.plot(arraystdev,a_err,'r--')
plt.plot(arraystdev,b_err,'ko',label='B error')
plt.plot(arraystdev,b_err,'k--')
plt.ylabel('Errors in estimates',fontsize=20)
plt.xlabel(r'$\sigma$',fontsize=20)
plt.title('Errors in estimates in linear scale plot',size=20)
plt.grid(True)
plt.legend()
plt.savefig('Linear.png')
plt.show()

#Log Log scale plot of errors in estimate vs stdev
plt.figure(figsize=(10,6))
plt.plot(arraystdev,a_err,'ro',label='A error')
plt.plot(arraystdev,a_err,'r--')
plt.plot(arraystdev,b_err,'ko',label='B error')
plt.plot(arraystdev,b_err,'k--')
plt.ylabel('Errors in estimates',fontsize=20)
plt.xlabel(r'$\sigma$',fontsize=20)
plt.yscale('log')
plt.xscale('log')
plt.title('Errors in estimates in log log scale plot',size=20)
plt.grid(True)
plt.legend()
plt.savefig('Loglog.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy

#Question 1 and 2
F1 = sp.lti([1,0.5],scipy.polyadd(scipy.polymul([1,0.5],[1,0.5]),[2.25]))

F2 = sp.lti([1,0.05],scipy.polyadd(scipy.polymul([1,0.05],[1,0.05]),[2.25]))

t = np.linspace(0,100,1001)
t,x_2 = sp.impulse(sp.lti(F2.num,scipy.polymul(F2.den,[1, 0, 2.25])),None,T=t)

t = np.linspace(0,100,1001)
t,x_1 = sp.impulse(sp.lti(F1.num,scipy.polymul(F1.den,[1, 0, 2.25])),None,T=t)
plt.figure(figsize=(10,7.5))
plt.title('Plots for Question 1 and 2',size=20)
plt.xlabel('t '+r'$\rightarrow$',size=20)
plt.ylabel('x(t) '+r'$\rightarrow$',size=20)
plt.plot(t,x_1,label='Damping=0.5')
plt.plot(t,x_2,label='Damping=0.05')
plt.legend(prop={'size':10})
plt.savefig('q1and2.png')
plt.show()

#Question 3:

#Transfer function definition:
H = sp.lti([1],[1,0,2.25])

def lti_system(H,f,t):
    t,x,svec = sp.lsim(H,f,t)
    return t,x

freq = np.arange(1.4,1.65,0.05)
t = np.linspace(0,200,1001)

Freq, Time = np.meshgrid(freq,t)

f = np.exp(-0.05*Time)*np.cos(Freq*Time)

steadystate = []
x = []

for i in range(len(freq)):
    t,x1 = lti_system(H,f[:,i],t)
    x2 = x1[int(0.9*len(x1)):-1]
    steadystate.append(x2.max())    #Find steady state value for resonance plotting
    x.append(x1)
    plt.figure(figsize=(10,7.5))
    plt.plot(t,x1)
    plt.title('System Response at frequency = %f' %freq[i],size = 20)
    plt.xlabel('t '+r'$\rightarrow$',size=20)
    plt.ylabel('x(t) '+r'$\rightarrow$',size = 20)
    plt.savefig('frequency_%f.png' %freq[i])
    plt.show()

x = np.transpose(x) #Final matrix containing outputs for different frequencies

plt.figure(figsize=(10,7.5))
plt.plot(freq,steadystate,'ro')
plt.title('Steady State values as a function of frequency',size=20)
plt.xlabel('t '+r'$\rightarrow$',size=20)
plt.ylabel('Steady State Amplitudes '+r'$\rightarrow$',size=20)
#plt.savefig('steadystatefreqdep.png')
plt.show()

#Question 4:
H_X = sp.lti([1],[1,0,3,0,0])
H_Y = sp.lti([1],[1,0,3,0,0])
t = np.linspace(0,50,500)
t,x, svec = sp.lsim(H_X,0,t,X0 = [0,-1,0,1])
t,y, svec = sp.lsim(H_Y,0,t,X0 = [0,2,0,0])
plt.figure(figsize=(10,7.5))
plt.title('Solution of coupled differential equations',size=20)
plt.plot(t,x,label='x(t)')
plt.plot(t,y,label='y(t)')
plt.xlabel('t '+r'$\rightarrow$',size=20)
plt.ylabel('Outputs '+r'$\rightarrow$',size=20)
plt.legend(prop={'size':20})
plt.savefig('Coupled.png')
plt.show()

#Question 5:
Hlowpass = sp.lti([1],[1e-12,1e-4,1])
w, S, phi = Hlowpass.bode()

plt.figure(figsize=(10,7.5))
plt.plot(w,S)
plt.xscale('log')
plt.title('Magnitude Response',size=20)
plt.xlabel(r'$\omega$'+' '+r'$\rightarrow$',size=20)
plt.ylabel('Magnitude Response '+r'$\rightarrow$',size=20)
plt.savefig('Magnituderesp.png')
plt.show()

plt.figure(figsize=(10,7.5))
plt.plot(w,phi)
plt.xscale('log')
plt.title('Phase Response',size=20)
plt.xlabel(r'$\omega$'+' '+r'$\rightarrow$',size=20)
plt.ylabel('Phase Response '+r'$\rightarrow$',size=20)
plt.savefig('Phaseresp.png')
plt.show()

t = np.arange(0,0.03,1e-7)
v_input = np.cos(1e3*t)-np.cos(1e6*t)
t, v_output, svec = sp.lsim(Hlowpass,v_input,t,X0=None)

plt.figure(figsize=(10,7.5))
plt.plot(t,v_output)
plt.title('Output at longer timescales')
plt.xlabel('t '+r'$\rightarrow$',size=20)
plt.ylabel(r'$v_{0}(t) \rightarrow$',size=20)
plt.savefig('zoomout.png')
plt.show()

plt.figure(figsize=(10,7.5))
plt.plot(t[0:1000],v_output[0:1000])
plt.title('Output at shorter timescales')
plt.xlabel('t '+r'$\rightarrow$',size=20)
plt.ylabel(r'$v_{0}(t) \rightarrow$',size=20)
plt.savefig('zoomin.png')
plt.show()

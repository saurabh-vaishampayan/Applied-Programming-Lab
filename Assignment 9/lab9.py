import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Question 1: Examples

def fftcompute(x,Tmax):
    dt = Tmax/len(x)
    w = np.linspace(-np.pi/dt,np.pi/dt,len(x)+1)[:-1]
    X = (1/len(x))*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))
    return w, X

def fftplot(w,X,filename,xlim):
    fig, ax = plt.subplots(2,1,figsize=(15,10))
    
    ax[0].plot(w,np.abs(X))
    if(xlim!=None):
        ax[0].set_xlim(xlim)
    ax[0].set_title('Magnitude Plot')
    ax[0].set_xlabel('w')
    
    phase = np.angle(X)
    phase[np.where(np.abs(X)<1e-2)]=0
    
    ax[1].plot(w,phase,'ro')
    if(xlim!=None):
        ax[0].set_xlim(xlim)
    ax[1].set_xlabel('w')
    ax[1].set_title('Phase Plot')
    fig.savefig(filename+'.png')
    plt.show()

def hamming(n):
    wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*np.arange(n)/(n-1)))
    return wnd

N = 64
Tmax = 2*np.pi
t = np.linspace(-Tmax/2,Tmax/2,N+1)[:-1]
x = np.sin(np.sqrt(2)*t)

w, X = fftcompute(x,Tmax)
fftplot(w,X,'sqrt2sin',[-10,10])

t2 = np.linspace(-3*np.pi,-np.pi,N+1)[:-1]
t3 = np.linspace(np.pi,3*np.pi,N+1)[:-1]
plt.plot(t,x,'bo')
plt.plot(t2,x,'ro')
plt.plot(t3,x,'ro')
plt.savefig('1.png')
plt.show()


y = x*hamming(N)
w, Y = fftcompute(y, Tmax)
fftplot(w,Y,'hammingsqrt2_1',[-10,10])
plt.plot(t,y,'bo')
plt.plot(t2,y,'ro')
plt.plot(t3,y,'ro')
plt.savefig('2.png')
plt.show()

N = 256
Tmax = 8*np.pi
t = np.linspace(-Tmax/2,Tmax/2,N+1)[:-1]
x = np.sin(np.sqrt(2)*t)
y = x*hamming(N)

w, Y = fftcompute(y,Tmax)
fftplot(w,Y,'hammingsqrt2_2',[-10,10])

#Question 2:
N = 512
Tmax = 8*np.pi
t = np.linspace(-Tmax/2,Tmax/2,N+1)[:-1]
x = (np.cos(0.86*t))**3
w, X = fftcompute(x,Tmax)
fftplot(w,X,'q2withouthamming',[-10,10])

y = x*hamming(N)
w, Y = fftcompute(y,Tmax)
fftplot(w,Y,'q2withhamming',[-10,10])

#Question 3:

#Estimator for frequency and phase:
def estimator(w,X):
    mag = np.abs(X)
    maxi = np.max(mag)
    mag[np.where(np.abs(X )<1e-3*maxi)]=0
    w_est = np.sum(np.abs(w)*mag**2)/np.sum(mag**2)
    w_err = np.sqrt((np.sum((np.abs(np.abs(w)-w_est)**2)*mag**2))/np.sum(mag**2))
    w_est_arrindex = int(w_est/(w[1]-w[0])+len(w)/2)
    phase_est = np.angle(X[w_est_arrindex])
    w_err_arrindex = int(w_err/(w[1]-w[0]))
    X_temp = X[w_est_arrindex-1-w_err_arrindex:w_est_arrindex+2+w_err_arrindex]
    phase_err = np.sqrt(np.sum(np.abs((np.angle(X_temp)-phase_est)*(X_temp)**2))/np.sum(np.abs(X_temp)**2))
    return w_est, w_err, phase_est, phase_err

N = 256
Tmax = 2*np.pi

w_err = 1
w_est = 1
while(w_err>0.25*w_est):
    t = np.linspace(-Tmax/2,Tmax/2,N+1)[:-1]
    x = np.cos(0.6*t+0.3)+0.1*np.random.randn(N)
    w, X = fftcompute(x*hamming(N),Tmax)
    w_est, w_err, phase_est, phase_err = estimator(w,X)
    N = 2*N
    Tmax = 2*Tmax
    
print(w_est)
print(w_err)
print(phase_est)
print(phase_err)
fftplot(w,X,'noisy',[-5,5])


N = 1024
wind = 64
Tmax = 2*np.pi
t = np.linspace(-Tmax/2,Tmax/2,N+1)[:-1]
dt = t[1]-t[0]
x = np.cos(16*t*(1.5+0.5*t/np.pi))
#Without Hamming
w, X = fftcompute(x,Tmax)
fftplot(w, X,'noslicenohamm',None)
#With Hamming
y = x*hamming(len(x))
w, X = fftcompute(y,Tmax)
fftplot(w, X,'nosliceyeshamm',None)

#Question 6:
x_slice = np.zeros((N-wind,wind))
y_slice = np.zeros((N-wind,wind))
X = np.zeros((N-wind,wind))
Y = np.zeros((N-wind,wind))

for i in range(N-wind):
    x_slice[i] = x[i:i+wind]
    y_slice[i] = x_slice[i]*hamming(wind)
    X[i] = (1/wind)*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_slice[i]))).real
    Y[i] = (1/wind)*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y_slice[i]))).real

w = np.linspace(-np.pi/dt,np.pi/dt,wind+1)[:-1]
time = np.linspace(-Tmax,Tmax-wind*Tmax/N,N-wind+1)[:-1]


fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')

W, Time = np.meshgrid(w, time)

ax.plot_surface(W, Time, np.abs(X), cmap='viridis', edgecolor='none')
ax.set_title('Surface plot of |X(t)|')
ax.set_xlabel('w')
ax.set_ylabel('Time')
plt.savefig('surfnohamm.png')
plt.show()

fig2 = plt.figure(figsize=(15,10))
ax2 = plt.axes(projection='3d')

ax2.plot_surface(W, Time, np.abs(Y), cmap='viridis', edgecolor='none')
ax2.set_title('Surface plot of |Y(t)|')
ax2.set_xlabel('w')
ax2.set_ylabel('Time')
plt.savefig('surfyeshamm.png')
plt.show()


fig, ax = plt.subplots(2,1,figsize=(15,10))
cf1 = ax[0].contourf(w,time,np.abs(X))
ax[0].set_xlabel('w')
ax[0].set_ylabel('Time')
fig.colorbar(cf1, ax=ax[0])

cf2 = ax[1].contourf(w, time, np.angle(X))
ax[1].set_xlabel('w')
ax[1].set_ylabel('Time')
fig.colorbar(cf2, ax=ax[1])

fig.savefig('contyesslicenohamm.png')
plt.show()

fig, ax = plt.subplots(2,1,figsize=(15,10))
cf1 = ax[0].contourf(w,time,np.abs(Y))
ax[0].set_xlabel('w')
ax[0].set_ylabel('Time')
fig.colorbar(cf1, ax=ax[0])

cf2 = ax[1].contourf(w, time, np.angle(Y))
ax[1].set_xlabel('w')
ax[1].set_ylabel('Time')
fig.colorbar(cf2, ax=ax[1])

fig.savefig('contyessliceyeshamm.png')
plt.show()
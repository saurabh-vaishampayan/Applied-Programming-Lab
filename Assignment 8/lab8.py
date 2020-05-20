import numpy as np
import matplotlib.pyplot as plt

#Question 1: Working out examples-

#FFT of a random sequence:
x = np.random.rand(8)
X = np.fft.fft(x)
y = np.fft.ifft(X)
print(np.c_[x,y])
print(np.abs(x-y).max())

#FFT of sin(5t):

N = 128
t = np.linspace(0, 2*np.pi, N)
x = np.sin(5*t)

X = np.fft.fft(x)
fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(np.abs(X))
ax[0].set_xlabel('k', size=15)
ax[0].set_ylabel('Magnitude', size=15)
ax[1].plot(np.unwrap(np.angle(X)))
ax[1].set_xlabel('k',size=15)
ax[1].set_ylabel('Phase',size=15)
fig.suptitle('Spectrum of sin(5t)',size=20)
fig.savefig('sin5t_1.png')
plt.show()


N = 128
t = np.linspace(0,2*np.pi,N+1)
t = t[:-1]
x = np.sin(5*t)
X = (1/N)*np.fft.fftshift(np.fft.fft(x))
w = np.linspace(-N/2,N/2,N+1)[:-1]
fig, ax = plt.subplots(2,1,figsize=(10,10))

ax[0].plot(w, np.abs(X))
ax[0].set_xlabel('k', size=15)
ax[0].set_ylabel('Magnitude', size=15)
ax[0].set_xlim([-10,10])

ax[1].plot(w, np.angle(X),'ro')
ii = np.where(np.abs(X)>1e-3)
ax[1].plot(w[ii], np.angle(X[ii]),'go')
ax[1].set_xlabel('k',size=15)
ax[1].set_ylabel('Phase',size=15)
ax[1].set_xlim([-10,10])
fig.suptitle('Spectrum of sin(5t)',size=20)
fig.savefig('sin5t_2.png')
plt.show()

def fftcompute(x,Twindow):
    N = len(x)
    X = (1/N)*np.fft.fftshift(np.fft.fft(x))
    
    w_int = np.pi/Twindow
    w = w_int*np.linspace(-N,N,N+1)[:-1]
    return w, X

def fftplot(w,X,funcname,figtitle,tol,xlimit):
    
    fig, ax = plt.subplots(2,1,figsize=(10,10))
    
    mag = np.abs(X)
    
    ax[0].plot(w, mag)
    ax[0].set_xlabel('w', size=15)
    ax[0].set_ylabel('Magnitude', size=15)
    if(xlimit!=None):
        ax[0].set_xlim(xlimit)
    
    phase = np.angle(X)
    phase[np.where(mag<tol)]=0
    phase[np.where(np.abs(phase)<tol)]=0
    ax[1].plot(w, phase,'ro')
    ax[1].set_xlabel('w',size=15)
    ax[1].set_ylabel('Phase',size=15)
    if(xlimit!=None):
        ax[1].set_xlim(xlimit)
    fig.suptitle('Spectrum of '+funcname,size=20)
    fig.savefig(figtitle+'.png')
    plt.show()
    
N = 128
t = np.linspace(0,2*np.pi,N+1)[:-1]
x = np.cos(10*t)+0.1*np.cos(10*t)*np.cos(t)
w, X = fftcompute(x,2*np.pi)
fftplot(w,X,'(1+0.1cos(t))cos(10t)','am1',1e-3,[-15,15])

N = 512
beg = -4*np.pi
end = 4*np.pi
Twind = end-beg
t = np.linspace(beg,end,N+1)[:-1]
x = np.cos(10*t)+0.1*np.cos(10*t)*np.cos(t)
w, X = fftcompute(x,Twind)
fftplot(w,X,'(1+0.1cos(t))cos(10t)','am2',1e-3,[-15,15])

x1 = (np.cos(t))**3
x2 = (np.sin(t))**3

w, X = fftcompute(x1, Twind)
fftplot(w,X,r'$cos^{3}(t)$','coscubed',1e-3,[-10,10])

w, X = fftcompute(x2, Twind)
fftplot(w,X,r'$sin^{3}(t)$','sincubed',1e-3,[-10,10])

x = np.cos(5*np.cos(t)+20*t)
w, X = fftcompute(x, Twind)
fftplot(w,X,r'$cos(20t+5cos(t))$','fm',1e-3,[-50,50])

T = 2*np.pi
N = 32
tol = 1e-6
err = 1+tol

t = np.linspace(-T/2,T/2,N+1)[:-1]
x = np.exp(-t**2/2)
Xold = (T/N)*np.fft.fft(np.fft.ifftshift(x))

while(err>tol):

    T = 2*T
    N = 2*N

    t = np.linspace(-T/2,T/2,N+1)[:-1]
    x = np.exp(-t**2/2)
    
    Xnew = (T/N)*(np.fft.fft(np.fft.ifftshift(x)))
    
    err = np.sum(np.abs(Xnew[::2]-Xold))
    
    Xold = Xnew

print(err)
w = np.linspace(-np.pi*N/T,np.pi*N/T,N+1)[:-1]
Xo = (np.sqrt(2*np.pi))*np.exp(-w**2/2)
Xnew = np.fft.fftshift(Xnew)
fftplot(w, Xnew, 'Fourier Transform of gaussian:Calculated','gaussian',1e-6,None)

plt.plot(w,np.sqrt(2*np.pi)*np.exp(-w**2/2))
plt.title('Fourier Transform of gaussian:Expected')
plt.xlabel('w')
plt.ylabel('|X|')
plt.savefig('gaussexp.png')
plt.show()

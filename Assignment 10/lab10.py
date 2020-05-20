import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import csv

def extract(filename):
    h = []
    with open(filename,'r') as f:
        data  = csv.reader(f)
        for row in data:
            temp = row[0].split('i')
            if(len(temp)==1):
                h.append(float(temp[0]))
            if(len(temp)==2):
                temp = temp[0].split('+')
                if(len(temp)==2):
                    h.append(float(temp[0])+1j*float(temp[1].split('i')[0]))
                else:
                    temp = temp[0].split('-')
                    if(len(temp)==2):
                        h.append(float(temp[0])-1j*float(temp[1].split('i')[0]))
                    else:
                        h.append(float(temp[1])-1j*float(temp[2].split('i')[0]))
    h = np.array(h)
    return h
  
h = extract('h.csv')
w, htransf = sp.freqz(h)
ii = np.where(np.abs(np.abs(htransf)-0.5)==np.min(np.abs(np.abs(htransf)-0.5)))
print(w[ii]) #3db Freq

zeros = np.roots(h)
x = np.real(zeros)
y = np.imag(zeros)
print(zeros)

theta = np.linspace(0,2*np.pi,101)[:-1]

X = np.cos(theta)
Y = np.sin(theta)

plt.figure(figsize=(10,6))
plt.plot(x,y,'ro')
plt.plot(X,Y,'go')
plt.plot(0,0,'bo')
plt.title('Zeros of h',size=20)
plt.savefig('zeros.png')
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(15,10))
ax[0].plot(w, np.abs(htransf))
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('w'+r'$\rightarrow$',fontsize=15)
ax[0].set_ylabel('|H|'+r'$\rightarrow$',fontsize=15)
    
ax[1].plot(w, np.unwrap(np.angle(htransf)))
ax[1].set_xscale('log')
ax[1].set_xlabel('w'+r'$\rightarrow$',fontsize=15)
ax[1].set_ylabel(r'$\phi$'+r'$\rightarrow$',fontsize=15)
fig.savefig('h.png')
plt.show()

n = np.arange(1, (2**10)+1, 1)
x = np.cos(0.2*np.pi*n)+np.cos(0.85*np.pi*n)
plt.figure(figsize=(15,10))
plt.plot(x)
plt.title('Input to filter', size=20)
plt.savefig('x.png')
plt.show()

y = np.convolve(x, h)
plt.figure(figsize=(15,10))
plt.plot(y)
plt.title('Output of linear convolution',size=20)
plt.savefig('linout.png')
plt.show()

w, X = sp.freqz(x)
plt.figure(figsize=(15,10))
plt.plot(w, np.abs(X))
plt.title('FFT of x',size=20)
plt.savefig('fftx.png')
plt.show()
w, Y = sp.freqz(y)
plt.figure(figsize=(15,10))
plt.plot(w, np.abs(Y))
plt.title('FFT of y',size=20)
plt.savefig('ffty.png')
plt.show()

y1 = np.fft.ifft(np.fft.fft(x)*np.fft.fft(np.concatenate((h,np.zeros(len(x)-len(h))))))
plt.figure(figsize=(15,10))
plt.plot(y1)
plt.title('Output of circular convoltion', size=20)
plt.savefig('circout.png')
plt.show()

y2 = np.concatenate((y1, np.zeros(len(y)-len(y1))))
plt.figure(figsize=(15,10))
plt.plot(y2-y)
plt.title('Difference in output by the two methods', size=15)
plt.savefig('diff.png')
plt.show()

#Linear convolution using circular convolution:
def overlap_add_conv(x,h):
    
    #Zero pad h to have length as power of 2
    M = len(h)
    n_ = int(np.ceil(np.log2(M)))
    N = 2**n_
    L = N+1-M
    h_ = np.concatenate((h,np.zeros(L-1)))
    
    #Make slices of x which are L units long
    Nx = len(x)
    n_slices = int(np.ceil(Nx/L))
    x_ = np.concatenate((x,np.zeros(L*n_slices-Nx)))
    
    y = np.zeros(len(x_)+M-1)
    
    for i in range(n_slices):
        temp = np.concatenate((x_[i*L:(i+1)*L],np.zeros(M-1)))
        y[i*L:(i+1)*L+M-1] += np.fft.ifft(np.fft.fft(h_)*np.fft.fft(temp)).real
    return y

y3 = overlap_add_conv(x,h)
y = np.concatenate((y, np.zeros(len(y3)-len(y))))
plt.figure(figsize=(15,10))
plt.plot(y3)
plt.title('Linear convolution using overlap-add circular convolution', size=10)
plt.savefig('overlapadd.png')
plt.show()
plt.figure(figsize=(15,10))
plt.plot(y3-y)
plt.title('Difference in outputs',size=15)
plt.savefig('diff2.png')
plt.show()

#Question 6:
zc = extract('x1.csv')
z2 = np.roll(zc,5)

corr = np.fft.ifftshift(np.correlate(z2,zc,'full'))
plt.figure(figsize=(15,10))
plt.plot(np.abs(corr[0:20]))
plt.title('Autocorrelation of x1',size=20)
plt.savefig('x1.png')
plt.show()
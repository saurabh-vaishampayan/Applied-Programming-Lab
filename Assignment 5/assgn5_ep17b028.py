import numpy as np
import matplotlib.pyplot as plt
from cmd import Cmd
import scipy.optimize
from mpl_toolkits import mplot3d

Nx = 25
Ny = 25
radius = 0.35
Niter = 1500

#cmd Prompt(shell subroutine) was borrowed from https://coderwall.com/p/w78iva/give-your-python-program-a-shell-with-the-cmd-module
class MyPrompt(Cmd):
    print("Type take_input Nx Ny radius Niter")
    def do_take_input(self, args):
        """Takes as input Nx, Ny,radius,Niter"""
        args = args.split(" ")
        
        if (len(args)!=4):
            print('Insufficient arguments')
        else:
            Nx = int(args[0])
            Ny = int(args[1])
            radius = float(args[2])
            Niter = int(args[3])
            x,y,X,Y,phi = potential_calc(Nx,Ny,radius,Niter)
            Jx,Jy = current_calc(x,y,Nx,Ny,radius,Niter,phi,X,Y)
            temperature_calc(Nx,Ny,Niter,Jx,Jy,X,Y)
            return True
        
    def do_quit(self, args):
        """Quits the program."""
        print ("Quitting.")
        raise SystemExit

def linear_fit(x,a,b):
    return a*x+b

def potential_calc(Nx,Ny,radius,Niter):
    phi =np.zeros((Ny,Nx))
    x = np.linspace(-0.5,0.5,Nx)
    y = np.linspace(-0.5,0.5,Ny)
    
    Y,X = np.meshgrid(y,x)
    
    ii = np.where(X*X+Y*Y<=radius*radius)
    phi[ii] = 1.00
    
        
    plt.figure(figsize=(10,10))
    cp = plt.contour(Y,X,phi)
    plt.clabel(cp)
    plt.title('Contour Plot of inital potential',size=20)
    plt.savefig('Contourplotinitialpotential.png')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    errors = np.zeros(Niter)
    
    for k in range(Niter):
        oldphi = phi.copy()
        phi[1:-1,1:-1] = 0.25*(phi[0:-2,1:-1]+phi[2:,1:-1]+phi[1:-1,0:-2]+phi[1:-1,2:])
        phi[1:-1,0] = phi[1:-1,1]   #Left edge
        phi[1:-1,-1] = phi[1:-1,-2]    #Right edge
        phi[-1,1:-1] = phi[-2,1:-1] #Top edge
        phi[ii] = 1.0
        phi[-1,0] = phi[-2,0]
        phi[-1,-1] = phi[-2,-1]
        phi[-1,-1] = phi[-1,-2]
        errors[k] = np.abs(oldphi-phi).max()
        
    plt.figure(figsize=(10,10))
    plt.title(r'$\phi$',size=20)
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, X, phi, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    plt.title(r'$\phi$',size=20)
    plt.savefig('phi.png')
    plt.show()
    
    plt.plot(errors)
    plt.yscale('log')
    plt.title('Variation of errors wrt iterations(semilog)')
    plt.xlabel('No of iterations')
    plt.ylabel('Error')
    plt.savefig('Errorssemilog.png')
    plt.show()
    
    n = np.arange(1,Niter+1,1)
    param, cov = scipy.optimize.curve_fit(linear_fit,n,np.log(errors))
    print('A is',np.exp(param[0]))
    print('B is',param[1])
    return x,y,X,Y,phi

def current_calc(x,y,Nx,Ny,radius,Niter,phi,X,Y):
    #Steady state currents
    Jx = np.zeros((Ny,Nx))
    Jy = np.zeros((Ny,Nx))
    
    Jx[1:-1,1:-1] = 0.5*(phi[1:-1,0:-2]-phi[1:-1,2:])
    Jx[0:,0] = 0
    Jx[-1,1:-1] = phi[-1,0:-2]-phi[-1,2:]
    Jx[0:,-1] = 0
    Jx[0,:] = 0
    
    Jy[1:-1,1:-1] = 0.5*(phi[0:-2,1:-1]-phi[2:,1:-1])
    Jy[0:,0] = 0
    Jy[-1,0:] = 0
    Jy[0:,-1] = 0
    Jy[0,:] = phi[0,:]-phi[1,:]
    
    plt.quiver(x,y,Jx,Jy)
    plt.title('Current distribtion',size=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('current.png')
    plt.show()
    return Jx,Jy

def temperature_calc(Nx,Ny,Niter,Jx,Jy,X,Y):
    #Solving for temperature distribution
    T = np.zeros((Ny,Nx))   #Initiallise to zero. Later add 300K to everything
    
    for k in range(Niter):
        T[1:-1,1:-1] = 0.25*(T[0:-2,1:-1]+T[2:,1:-1]+T[1:-1,0:-2]+T[1:-1,2:])-0.5*(Jx[1:-1,1:-1]*Jx[1:-1,1:-1]+Jy[1:-1,1:-1]*Jy[1:-1,1:-1])
        T[-1,1:-1] = T[-2,1:-1]
        T[1:-1,0] = T[1:-1,1]
        T[1:-1,-1] = T[1:-1,-2]
        T[-1,-1] = T[-1,-2]
        T[-1,0] = T[-1,1]
    T = T+300
    
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, X, T, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    plt.title('Temperature Plot',size=20)
    plt.savefig('Temperaturesurface.png')
    plt.show()

    
    plt.figure(figsize=(10,10))
    cp = plt.contour(Y,X,T)
    plt.clabel(cp)
    plt.title('Temperature Contour Plot',size=20)
    plt.savefig('Temperaturecontour.png')
    plt.show()
    

if __name__ == '__main__':
    prompt = MyPrompt()
    prompt.prompt = '> '
    prompt.cmdloop('Starting prompt...')
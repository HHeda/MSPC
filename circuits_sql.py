import numpy as np
import warnings
import os
import qucat as qc
from scipy.constants import *
from scipy.optimize import minimize
import sympy as sp
from sympy.utilities.lambdify import lambdify
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import newton, root_scalar, newton_krylov, root, fsolve, fmin_powell,  fmin
from numpy.linalg import norm
import zlib
from numba import jit
from data_storage import *

#warnings.simplefilter('ignore')


fac = h/(e**2)
f0 = 1e6

phi_0 = hbar/2/e



def conv_L(L):
    return L*f0/fac

def conv_C(C):
    return C*fac*f0

def conv_R(R):
    return R/fac

def conv_freq(f):
    return f/f0

def conv_energ(E):
    return E/h/f0

def conv_inv_L(L):
    return L/f0*fac

def conv_inv_C(C):
    return C/fac/f0

def conv_inv_R(R):
    return R*fac

def conv_inv_freq(f):
    return f*f0

def conv_inv_energ(E):
    return E*h*f0

def Ic(E):
    phi0 = 0.5
    return 2*pi*E/phi0

a_in0 = np.sqrt(conv_freq(1/100e-9))

E = (1/4/pi)**2/(conv_L(3e-9))*2.2
#E = (hbar/2./e)**2/3e-9/h*2.2 #In actual unities


class SPC(object):
    """class manipulating a SPC (or MSPC) as a qucat circuit, saving the values obtained. 
    Does the heavy calculation only if they have never been done before.
    Lazy mode: doesn't calculates the circuit. This calculation will be triggered if a new value is needed
    (time needed: 10-20 min).
    """
    
    def __init__(self,alpha, E, N, name, default = False, edit = False, plot = True, res = False, lazy = True):
        '''
        E: Josephson energy of the junctions of the snails, in MHz
        N: Number of snails in each branch
        '''
        #if default : initializes a default SPC
        self.name = name
        self.default = default
        self.edit = edit
        self.plot = plot
        self.res = res
        self.lazy = lazy #tells if the circuit has to be calculated
        self.data_storage = Data("MSPC_res.db")
        self.data_storage.create()
        
        self.alpha = alpha
        self.N = N
        self.E = N * E
        phi = sp.Symbol('phi')
        phi_ext = sp.Symbol('phi_ext')
        self.H = - N*alpha*sp.cos(phi/N) - N*3*sp.cos((phi/N-phi_ext)/3)
        self.H1 = sp.diff(self.H, phi, 1)
        self.H2 = sp.diff(self.H, phi, 2)
        self.H3 = sp.diff(self.H, phi, 3)
        self.H4 = sp.diff(self.H, phi, 4)
        
        self.Hf = lambdify([phi, phi_ext], self.H)
        self.H1f = lambdify([phi, phi_ext], self.H1)
        self.H2f = lambdify([phi, phi_ext], self.H2)
        self.H3f = lambdify([phi, phi_ext], self.H3)
        self.H4f = lambdify([phi, phi_ext],self. H4)
        
        self.coeffs = np.vectorize(self.coeffs_nv)
        self.phi_ext = None
        
        if default and os.path.exists(name):
            if name[-4:] == ".txt":
                os.remove(name)
            else:
                return NameError("is not a .txt file")
        
        if default:
            #creates the file
            CyInt = 12.0e-14
            CxInt = 13.0e-14
            Cshunt = 5.0e-13
            CxExt = 4.5e-14
            CyExt = 3.0e-14
            R = 50
            if not res :
                lines = (
                """C;3,-8;3,-9;CyInts;CyInt
                    C;4,-8;5,-8;Cshunts;Cshunt
                    W;5,-8;6,-8;;
                    C;5,-8;5,-9;CyInts;CyInt
                    C;8,-8;9,-8;CxInts;CxInt
                    C;8,-7;8,-6;Cshunts;Cshunt
                    W;7,-8;8,-8;;
                    W;8,-8;8,-7;;
                    C;8,-5;9,-5;CxInts;CxInt
                    C;5,-3;5,-4;CyInts;CyInt
                    C;4,-5;5,-5;Cshunts;Cshunt
                    C;3,-4;3,-5;CyInts;CyInt
                    C;-1,-8;0,-8;CxInts;CxInt
                    C;0,-7;0,-6;Cshunts;Cshunt
                    W;0,-8;0,-7;;
                    W;0,-6;0,-5;;
                    W;0,-8;1,-8;;
                    W;0,-5;1,-5;;
                    W;2,-5;3,-5;;
                    W;3,-5;4,-5;;
                    W;5,-5;6,-5;;
                    W;7,-5;8,-5;;
                    G;5,-10;5,-9;;
                    G;-2,-8;-1,-8;;
                    G;3,-3;3,-4;;
                    G;5,-2;5,-3;;
                    G;10,-5;9,-5;;
                    G;10,-8;9,-8;;
                    W;8,-6;8,-5;;
                    G;3,-10;3,-9;;
                    D;1,-8;2,-8;;E1,E2,E3
                    D;6,-8;7,-8;;E1,E2,E3
                    D;6,-5;7,-5;;E1,E2,E3
                    D;1,-5;2,-5;;E1,E2,E3
                    W;5,-5;5,-4;;
                    W;2,-8;3,-8;;
                    W;4,-8;3,-8;;
                    C;0,-5;-1,-5;CxInts;Cxint
                    G;-2,-5;-1,-5;;""").replace(" ", "")

            else: 
                lines = (
                     """C;2,-9;2,-10;CyInts;CyInt
                        C;4,-9;5,-9;Cshunts;Cshunt
                        W;6,-9;7,-9;;
                        C;6,-9;6,-10;CyInts;CyInt
                        C;9,-9;10,-9;CxInts;CxInt
                        C;9,-7;9,-6;Cshunts;Cshunt
                        W;8,-9;9,-9;;
                        W;9,-9;9,-8;;
                        W;9,-8;9,-7;;
                        C;9,-5;10,-5;CxInts;CxInt
                        C;6,-3;6,-4;CyInts;CyInt
                        C;4,-5;5,-5;Cshunts;Cshunt
                        C;2,-4;2,-5;CyInts;CyInt
                        C;-1,-5;0,-5;CxInts;CxInt
                        C;-1,-9;0,-9;CxInts;CxInt
                        C;0,-7;0,-6;Cshunts;Cshunt
                        W;0,-9;0,-8;;
                        W;0,-8;0,-7;;
                        W;0,-6;0,-5;;
                        W;0,-9;1,-9;;
                        W;0,-5;1,-5;;
                        W;2,-5;3,-5;;
                        W;3,-5;4,-5;;
                        W;6,-5;7,-5;;
                        W;8,-5;9,-5;;
                        G;6,-11;6,-10;;
                        G;-2,-9;-1,-9;;
                        G;-2,-5;-1,-5;;
                        G;2,-3;2,-4;;
                        G;6,-2;6,-3;;
                        G;11,-5;10,-5;;
                        G;11,-9;10,-9;;
                        W;9,-6;9,-5;;
                        G;2,-11;2,-10;;
                        D;1,-9;2,-9;;E1,E2,E3
                        D;7,-9;8,-9;;E1,E2,E3
                        D;7,-5;8,-5;;E1,E2,E3
                        D;1,-5;2,-5;;E1,E2,E3
                        W;6,-5;6,-4;;
                        W;2,-9;3,-9;;
                        W;4,-9;3,-9;;
                        C;-1,-7;0,-7;CxExts;CxExts
                        C;-1,-6;0,-6;CxExts;CxExt
                        C;9,-7;10,-7;CxExts;CxExt
                        C;9,-6;10,-6;CxExts;CxExt
                        C;4,-9;4,-10;CyExts;CyExt
                        C;5,-9;5,-10;CyExts;CyExt
                        C;4,-4;4,-5;CyExts;CyExt
                        C;5,-4;5,-5;CyExts;CyExt
                        W;4,-10;5,-10;;
                        W;5,-9;6,-9;;
                        W;5,-5;6,-5;;
                        W;-1,-7;-1,-6;;
                        W;10,-7;10,-6;;
                        W;4,-4;5,-4;;
                        R;-2,-7;-1,-7;res;
                        R;10,-6;11,-6;res;
                        R;4,-11;4,-10;res;
                        R;4,-4;4,-3;res;
                        G;4,-2;4,-3;;
                        G;-3,-7;-2,-7;;
                        G;4,-12;4,-11;;
                        G;12,-6;11,-6;;""").replace(" ", "")

            lines = lines.replace("CxInts", str(conv_C(CxInt)))
            lines = lines.replace("CyInts", str(conv_C(CyInt)))
            lines = lines.replace("Cshunts", str(conv_C(Cshunt)))
            lines = lines.replace("CxExts", str(conv_C(CxExt)))
            lines = lines.replace("CyExts", str(conv_C(CyExt)))
            lines = lines.replace("res", str(conv_R(R)))
            lines = lines.split("\n")
                
            with open (name, "w") as f:
                for c in lines:
                    f.write(c+"\n")
                f.close
        
        self.sp = None #sweet spot
        if not self.lazy: #if we want the circuit to be calculated
            self.circuit = qc.GUI(name, edit = edit, plot = plot)
            
        
    def sweet_spot(self):
        if self.sp is None :
            def f(phi_ext):
                return self.coeffs_nv(phi_ext)[2]
            self.sp = root_scalar(f, bracket=[0, np.pi], xtol=1e-15).root
        return self.sp
        
    def return_kwargs(self, **kwargs):
        return kwargs
        
    def set_kwargs(self, E1, E2, E3):
        return self.return_kwargs(E1 = E1 , E2 = E2, E3 = E3)
    
    def redefine_param(self, alpha, E, N): #To avoid full recalculation of the circuit
        self.alpha = alpha
        self.E = N*E
        phi = sp.Symbol('phi')
        phi_ext = sp.Symbol('phi_ext')
        self.H = - N*alpha*sp.cos(phi/N) - N*3*sp.cos((phi/N-phi_ext)/3)
        self.H1 = sp.diff(self.H, phi, 1)
        self.H2 = sp.diff(self.H, phi, 2)
        self.H3 = sp.diff(self.H, phi, 3)
        self.H4 = sp.diff(self.H, phi, 4)
        
        self.Hf = lambdify([phi, phi_ext], self.H)
        self.H1f = lambdify([phi, phi_ext], self.H1)
        self.H2f = lambdify([phi, phi_ext], self.H2)
        self.H3f = lambdify([phi, phi_ext], self.H3)
        self.H4f = lambdify([phi, phi_ext],self. H4)
        self.sp = None
        
    def coeffs_nv(self, phi_ext): #gives the coeefficient corresponding to a phi_ext
        x0 = max(np.abs(self.H1f(0, phi_ext)), np.abs(self.H1f(1, phi_ext)))
        self.c2 = self.E*self.H2f(minimize(self.Hf, x0, args = (phi_ext,)).x, phi_ext)
        self.c3 =  self.E*self.H3f(minimize(self.Hf, x0, args = (phi_ext,)).x, phi_ext)
        self.c4 =  self.E*self.H4f(minimize(self.Hf, x0, args = (phi_ext,)).x, phi_ext)
        self.phi_ext = phi_ext
        return self.c2[0], self.c3[0], self.c4[0]
    
    
    def data(self, phi_ext, pretty_print = False):
        """Checks if this vector has already been saved. 
        If it is the case, loads the corresponding values. If not, calculates it.
        """
        try:
            len(phi_ext)
            vector = True
        except:
            vector = False
        if vector:
            key = zlib.adler32(np.array([self.alpha, self.E, self.N] + list(phi_ext), dtype = float))

            if self.res:
                name_dir = "data_res/" + str(self.N) + "/"
                try:
                    os.mkdir("data_res/")
                except:
                    pass

            else:
                name_dir = "data/"+ str(self.N) + "/"
                try:
                    os.mkdir("data/")
                except:
                    pass
            try:
                os.listdir(name_dir)
            except FileNotFoundError:
                os.mkdir(name_dir)

            if str(key) not in os.listdir(name_dir):
                if self.lazy:
                    print("Lazy mode and new phi_ext, calculating the circuit...")
                    try:
                        self.__init__(self.alpha, self.E/self.N, self.N, self.name,
                                      default = self.default, 
                                      edit = self.edit, 
                                      plot = self.plot, 
                                      res = self.res, 
                                      lazy = False)
                    except KeyboardInterrupt:
                        self.lazy = True
                        raise KeyboardInterrupt
                    print("Circuit calculated")

                E2s, E3s, E4s = self.coeffs(phi_ext)
                kwargs = self.set_kwargs(E2s, E3s, E4s)
                self.freqs, self.dissip, self.anh, self.chi = self.circuit.f_k_A_chi(**kwargs, pretty_print = pretty_print)
                self.freqs, self.dissip, self.anh, self.chi = (np.real(self.freqs[-4:]), 
                                                               np.real(self.dissip[-4:]), 
                                                               np.real(self.anh[-4:]), 
                                                               np.real(self.chi[-4:,-4:]))
                self.t_w = np.real(self.circuit.three_waves(-3, -2, -1, **kwargs))


                phi_zpf_0 = np.abs(self.circuit.junctions[0].zpf(0, 'flux', **kwargs))
                phi_zpf_1 = np.abs(self.circuit.junctions[0].zpf(1, 'flux', **kwargs))
                phi_zpf_2 = np.abs(self.circuit.junctions[0].zpf(2, 'flux', **kwargs))
                phi_zpf_3 = np.abs(self.circuit.junctions[0].zpf(3, 'flux', **kwargs))

                self.zpf = np.real(np.array([phi_zpf_0, phi_zpf_1, phi_zpf_2, phi_zpf_3]))
                self.phi_ext = phi_ext
                os.mkdir(name_dir+str(key))
                np.save(name_dir+ str(key)+"/phi_ext", np.array(self.phi_ext, dtype = float))
                np.save(name_dir+ str(key)+"/freqs", self.freqs)
                np.save(name_dir+ str(key)+"/dissip", self.dissip)
                np.save(name_dir+ str(key)+"/anh", self.anh)
                np.save(name_dir+ str(key)+"/chi", self.chi)
                np.save(name_dir+ str(key)+"/t_w", self.t_w)
                np.save(name_dir+ str(key)+"/zpf", self.zpf)
            else:
                self.phi_ext = phi_ext
                self.freqs = np.real(np.load(name_dir+ str(key)+"/freqs.npy"))
                self.dissip = np.real(np.load(name_dir+ str(key)+"/dissip.npy"))
                self.anh = np.real(np.load(name_dir+ str(key)+"/anh.npy"))
                self.chi = np.real(np.load(name_dir+ str(key)+"/chi.npy"))
                self.t_w = np.real(np.load(name_dir+ str(key)+"/t_w.npy"))
                self.zpf = np.real(np.load(name_dir+ str(key)+"/zpf.npy"))
            
            self.data_storage.save(key, self.alpha, self.E, self.N, self.phi_ext, 
                               self.freqs, 
                               self.dissip, 
                               self.anh, 
                               self.chi, 
                               self.t_w, 
                               self.zpf)
        else:
            key = zlib.adler32(np.array([self.alpha, self.E, self.N, phi_ext], dtype = float))

            try:
                (self.freqs, 
                self.dissip, 
                self.anh, 
                self.chi, 
                self.t_w, 
                self.zpf) = self.data_storage.extract(self.alpha, self.E, self.N, phi_ext)
                self.phi_ext = phi_ext
            except TypeError:
                print('alpha, E, N, phi_ext not in database')
                if self.lazy:
                    print("Lazy mode and new phi_ext, calculating the circuit...")
                    try:
                        self.__init__(self.alpha, self.E/self.N, self.N, self.name,
                                      default = self.default, 
                                      edit = self.edit, 
                                      plot = self.plot, 
                                      res = self.res, 
                                      lazy = False)
                    except KeyboardInterrupt:
                        self.lazy = True
                        raise KeyboardInterrupt
                    print("Circuit calculated")
                E2s, E3s, E4s = self.coeffs(phi_ext)
                kwargs = self.set_kwargs(E2s, E3s, E4s)
                self.freqs, self.dissip, self.anh, self.chi = self.circuit.f_k_A_chi(**kwargs, pretty_print = pretty_print)
                self.freqs, self.dissip, self.anh, self.chi = (np.real(self.freqs[-4:]), 
                                                               np.real(self.dissip[-4:]), 
                                                               np.real(self.anh[-4:]), 
                                                               np.real(self.chi[-4:,-4:]))
                self.t_w = np.real(self.circuit.three_waves(-3, -2, -1, **kwargs))



                phi_zpf_0 = np.abs(self.circuit.junctions[0].zpf(0, 'flux', **kwargs))
                phi_zpf_1 = np.abs(self.circuit.junctions[0].zpf(1, 'flux', **kwargs))
                phi_zpf_2 = np.abs(self.circuit.junctions[0].zpf(2, 'flux', **kwargs))
                phi_zpf_3 = np.abs(self.circuit.junctions[0].zpf(3, 'flux', **kwargs))

                self.zpf = np.real(np.array([phi_zpf_0, phi_zpf_1, phi_zpf_2, phi_zpf_3]))
                self.phi_ext = phi_ext
                self.data_storage.save(key, self.alpha, self.E, self.N, self.phi_ext, 
                                       self.freqs, 
                                       self.dissip, 
                                       self.anh, 
                                       self.chi, 
                                       self.t_w, 
                                       self.zpf)



                
                
    def eigenfrequencies(self, phi_ext): #circuit.eigenfrequencies with the corresponding values for phi_ext
        self.data(phi_ext)
        return self.freqs
            
    
    def anharmonicities(self, phi_ext): #circuit.anharmonicities with the corresponding values for phi_ext
        self.data(phi_ext)
        return self.anh
    
    def show_normal_mode(self, i, phi_ext, quantity = 'current'): #Mode zero always unphysical
        E2s, E3s, E4s = self.coeffs_nv(phi_ext)
        kwargs = self.set_kwargs(E2s, E3s, E4s)
        try:
            self.circuit
        except AttributeError:
            print("Lazy mode, calculating the circuit...")
            try:
                self.__init__(self.alpha, self.E/self.N, self.N, self.name,
                              default = self.default, 
                              edit = self.edit, 
                              plot = self.plot, 
                              res = self.res, 
                              lazy = False)
            except KeyboardInterrupt:
                self.lazy = True
                raise KeyboardInterrupt
            print("Circuit calculated")

        self.data(phi_ext)
        return self.circuit.show_normal_mode(i, **kwargs, quantity = quantity)
    
        
    def f_k_A_chi(self, phi_ext, pretty_print = False):
        self.data(phi_ext, pretty_print = pretty_print)

        return self.freqs, self.dissip, self.anh, self.chi
    
    def three_waves_term(self, phi_ext):
        self.data(phi_ext)
        return self.t_w

    
    def coupling_rates(self, phi_ext):
        self.data(phi_ext)
        return self.dissip
    
    def phis_zpf(self, phi_ext):
        self.data(phi_ext) 
        return self.zpf
        
    def plot_eigenfrequencies(self ,n): #plots the eigenfrequences for phi_ext in 0, 6pi, with n points
        phi_exts = np.linspace(0, 6*pi, n)
        self.data(phi_exts)
        
        plt.figure("eigenfrequencies")
        plt.plot(phi_exts/pi, self.freqs[0]*f0/1e9, label = "mode 0")
        plt.plot(phi_exts/pi, self.freqs[1]*f0/1e9, label = "mode 1")
        plt.plot(phi_exts/pi, self.freqs[2]*f0/1e9, label = "mode 2")
        plt.plot(phi_exts/pi, self.freqs[3]*f0/1e9, label = "mode 3")
        plt.xlabel('$\phi_{ext}/\pi$')
        plt.ylabel('Normal mode frequency (GHz)')
        plt.legend()
        plt.show()
        
    def plot_anharmonicities(self, n): 
        phi_exts = np.linspace(0, 6*pi, n)
        self.data(phi_exts)
        
        plt.figure("anharmonicities")
        plt.plot(phi_exts/pi, self.anh[0]*f0/1e6, label = "mode 0")
        plt.plot(phi_exts/pi, self.anh[1]*f0/1e6, label = "mode 1")
        plt.plot(phi_exts/pi, self.anh[2]*f0/1e6, label = "mode 2")
        plt.plot(phi_exts/pi, self.anh[3]*f0/1e6, label = "mode 3")
        plt.xlabel('$\phi_{ext}/\pi$')
        plt.ylabel('Mode anharmonicity (MHz)')
        plt.legend()
        plt.show()
        

        
    def plot_three_waves_term(self,n):
        phi_exts = np.linspace(0, 6*pi, n)
        self.data(phi_exts)
        plt.figure("Three-wave-mixing term")
        plt.plot(phi_exts/pi, self.t_w*f0/1e6)
        plt.xlabel('$\phi_{ext}/\pi$')
        plt.ylabel('1-2-3 wave mixing (MHz)')
        plt.show()

        
    def plot_coupling_rates(self,n):
        phi_exts = np.linspace(0, 6*pi, n)
        self.data(phi_exts)
        plt.figure("coupling rates")
        plt.plot(phi_exts/pi, self.dissip[0]*f0/1e6, label = "mode 0")
        plt.plot(phi_exts/pi, self.dissip[1]*f0/1e6, label = "mode 1")
        plt.plot(phi_exts/pi, self.dissip[2]*f0/1e6, label = "mode 2")
        plt.plot(phi_exts/pi, self.dissip[3]*f0/1e6, label = "mode 3")        
        plt.xlabel('$\phi_{ext}/\pi$')
        plt.ylabel('coupling rates (MHz)')
        plt.show()




    
################### Functions : gain, bandwidth... #######################
def scattering_matrix_f(s, phi_ext, p0, f1, fp): #without the self-Kerr
    Np = np.abs(p0)**2
    omega_1 = 2*np.pi*f1
    omega_2 = 2*np.pi*(fp-f1)
    freq, k, A, chi = s.f_k_A_chi(phi_ext)
    kappas = s.coupling_rates(phi_ext)
    omega_a = 2*pi*(freq[1] 
                  - A[1] 
                  - (chi[0, 1] + chi[2, 1] + chi[3, 1])/2
                  - chi[3, 1]*Np)
    omega_b = 2*pi*(freq[2] 
                  - A[2] 
                  - (chi[0, 2] + chi[1, 2] + chi[3, 2])/2
                  - chi[3, 2]*Np)
    Delta1, Delta2 = omega_1 - omega_a, omega_2-omega_b
    delta1, delta2 = 2*Delta1/kappas[1], 2*Delta2/kappas[2]
    rho = 2*s.three_waves_term(phi_ext)*p0/np.sqrt(kappas[1]*kappas[2])
    rho2 = np.abs(rho)**2
    mat = np.zeros((2, 2, len(f1)), dtype = complex)
    a = (((1-1j*delta2)*(1-1j*delta1) + rho2)/
                        ((1-1j*delta2)*(1+1j*delta1)-rho2))
    b = (((1+1j*delta2)*(1+1j*delta1) + rho2)/
                        ((1-1j*delta2)*(1+1j*delta1)-rho2))
    c = 2*1j*rho/((1-1j*delta2)*(1+1j*delta1)-rho2)

    d = -2*1j*np.conj(rho)/((1-1j*delta2)*(1+1j*delta1)-rho2)
    mat[0][0] = a
    mat[1][1] = b
    mat[0][1] = c
    mat[1][0] = d
    return mat


def scattering_matrix_sp(s, phi_ext, p0, f1, fp): #at the sweet spot
    N_pump = np.abs(p0)**2
    freqs, k, A, chi = s.f_k_A_chi(phi_ext)
    kappas = s.coupling_rates(phi_ext)
    Delta1, Delta2 = (2*pi*(f1-freqs[1]), 2*pi*(fp-f1-freqs[2]))

    delta1, delta2 = 2*Delta1/kappas[1], 2*Delta2/kappas[2]
    rho = 2*s.three_waves_term(phi_ext)*p0/np.sqrt(kappas[1]*kappas[2])
    rho2 = np.abs(rho)**2
    mat = np.zeros((2, 2, len(f1)), dtype = complex)
    a = (((1-1j*delta2)*(1-1j*delta1) + rho2)/
                        ((1-1j*delta2)*(1+1j*delta1)-rho2))
    b = (((1+1j*delta2)*(1+1j*delta1) + rho2)/
                        ((1-1j*delta2)*(1+1j*delta1)-rho2))
    c = 2*1j*rho/((1-1j*delta2)*(1+1j*delta1)-rho2)

    d = -2*1j*np.conj(rho)/((1-1j*delta2)*(1+1j*delta1)-rho2)
    mat[0][0] = a
    mat[1][1] = b
    mat[0][1] = c
    mat[1][0] = d
    return mat


def gain_f(s, phi_ext, p0, f1, fp):
    return np.abs(scattering_matrix_f(s, phi_ext, p0, f1, fp)[0][0])**2

def gain_sp(s, phi_ext, p0, f1, fp):
    return np.abs(scattering_matrix_sp(s, phi_ext, p0, f1, fp)[0][0])**2

def bandwidth_sp(s, phi_ext, p0, f1, fp):
    return gain_bandwith_sp()/np.sqrt(self.gain_sp(s, phi_ext, p0, f1, fp))


def gain_bandwith_sp(s, phi_ext):
    kappas = s.coupling_rates(phi_ext)
    return 2/(1/kappas[1]+1/kappas[2])

def solve_stat(s, phi_ext, a_in, b_in, p0, f1, fp):
    Np = np.abs(p0)**2
    omega_1 = 2*np.pi*f1
    omega_2 = 2*np.pi*(fp-f1)
    freq, k, A, chi = s.f_k_A_chi(phi_ext)


    kappas = s.coupling_rates(phi_ext).reshape((-1,))
    omega_a = 2*pi*(freq[1] 
                  - 3/2*A[1] 
                  - (chi[0, 1] + chi[2, 1] + chi[3, 1])/2
                  - chi[3, 1]*Np)
    omega_b = 2*pi*(freq[2] 
                  - 3/2*A[2] 
                  - (chi[0, 2] + chi[1, 2] + chi[3, 2])/2
                  - chi[3, 2]*Np)


    chim = s.three_waves_term(phi_ext)
    vect0 = np.array([np.real(a_in)/np.sqrt(kappas[1]), 
                        np.imag(a_in),
                        np.real(b_in),
                        np.imag(b_in)])
    def optimizer(vect):
        a,b = vect[0] + 1j*vect[1], vect[2] + 1j*vect[3]
        c_res =  (1j*(omega_1 - omega_a + A[1]*np.abs(a)**2)*a
                -1j * chim*p0*np.conj(b)
                - kappas[1]/2*a
                + np.sqrt(kappas[1])*a_in, 
                1j*(omega_2 - omega_b + A[2]/2*np.abs(b)**2)*b
                -1j * chim*p0*np.conj(a)
                - kappas[2]/2*b
                + np.sqrt(kappas[2])*b_in)
        return np.array([np.real(c_res[0]), 
                        np.imag(c_res[0]),
                        np.real(c_res[1]),
                        np.imag(c_res[1])])
    opti = root(optimizer, vect0, method = "lm", tol = 1e-15)
    sol = opti.x
    success = opti.success
    if not success:
        print("did not converge for a_in = ", a_in)
    a_stat = sol[0] + 1j * sol[1]
    return (-a_in + np.sqrt(kappas[1])*a_stat)/a_in

@jit
def optimizer(vect, p0, delta_1, delta_2, A, chim, kappas, a_inr, a_ini, b_inr, b_ini):
    a2 = vect[0]**2 + vect[1]**2
    b2 = vect[2]**2 + vect[3]**2
    res =  np.array([-(delta_1 - A[1]/2*a2)*vect[1]
                    - chim*p0*vect[3]
                    - kappas[1]/2*vect[0]
                    + np.sqrt(kappas[1])*a_inr,
                    +(delta_1 - A[1]/2*a2)*vect[0]
                    - chim*p0*vect[2]
                    - kappas[1]/2*vect[1]
                    + np.sqrt(kappas[1])*a_ini,
                    -(delta_2 - A[2]/2*b2)*vect[3]
                    - chim*p0*vect[1]
                    - kappas[2]/2*vect[2]
                    + np.sqrt(kappas[2])*b_inr,
                    +(delta_2 - A[2]/2*b2)*vect[2]
                    - chim*p0*vect[0]
                    - kappas[2]/2*vect[3]
                    + np.sqrt(kappas[2])*b_ini])
    return res

def solve_stat3(s, phi_ext, a_in, b_in, p0, f1, fp):
    Np = np.abs(p0)**2
    omega_1 = 2*np.pi*f1
    omega_2 = 2*np.pi*(fp-f1)
    freq, k, A, chi = s.f_k_A_chi(phi_ext)
    kappas = s.coupling_rates(phi_ext)
    omega_a = 2*pi*(freq[1] 
                  - A[1] 
                  - (chi[0, 1] + chi[2, 1] + chi[3, 1])/2
                  - chi[3, 1]*Np)
    omega_b = 2*pi*(freq[2] 
                  - A[2] 
                  - (chi[0, 2] + chi[1, 2] + chi[3, 2])/2
                  - chi[3, 2]*Np)


    chim = s.three_waves_term(phi_ext)
    vect0 = np.array([np.real(a_in), 
                        np.imag(a_in),
                        np.real(b_in),
                        np.imag(b_in)])
    a_inr = np.real(a_in)
    a_ini = np.imag(a_in)
    b_inr = np.real(b_in)
    b_ini = np.imag(b_in)
    delta_1, delta_2 = omega_1-omega_a, omega_2 - omega_b
                          
    sol = fsolve(optimizer, vect0, args = (p0, delta_1, delta_2, A, chim, kappas, a_inr, a_ini, b_inr, b_ini))
    a_stat = sol[0] + 1j * sol[1]
    return (-a_in + np.sqrt(kappas[1])*a_stat)/a_in

def solve_stat2(s, phi_ext, a_in, b_in, p0, f1, fp):
    Np = np.abs(p0)**2
    omega_1 = 2*np.pi*f1
    omega_2 = 2*np.pi*(fp-f1)
    freq, k, A, chi = s.f_k_A_chi(phi_ext)
    kappas = s.coupling_rates(phi_ext)
    omega_a = 2*pi*(freq[1] 
                  - A[1] 
                  - (chi[0, 1] + chi[2, 1] + chi[3, 1])/2
                  - chi[3, 1]*Np)
    omega_b = 2*pi*(freq[2] 
                  - A[2] 
                  - (chi[0, 2] + chi[1, 2] + chi[3, 2])/2
                  - chi[3, 2]*Np)

    delta_1, delta_2 = omega_1-omega_a, omega_2 - omega_b

    chim = s.three_waves_term(phi_ext)
    vect0 = np.array([np.real(a_in), 
                        np.imag(a_in),
                        np.real(b_in),
                        np.imag(b_in)])
    a_inr = np.real(a_in)
    a_ini = np.imag(a_in)
    b_inr = np.real(b_in)
    b_ini = np.imag(b_in)
    def optimizer(vect):
        a2 = vect[0]**2 + vect[1]**2
        b2 = vect[2]**2 + vect[3]**2
        res =  np.array([-(omega_1 - omega_a - A[1]/2*a2)*vect[1]
                        - chim*p0*vect[3]
                        - kappas[1]/2*vect[0]
                        + np.sqrt(kappas[1])*a_inr,
                        +(omega_1 - omega_a - A[1]/2*a2)*vect[0]
                        - chim*p0*vect[2]
                        - kappas[1]/2*vect[1]
                        + np.sqrt(kappas[1])*a_ini,
                        -(omega_2 - omega_b - A[2]/2*b2)*vect[3]
                        - chim*p0*vect[1]
                        - kappas[2]/2*vect[2]
                        + np.sqrt(kappas[2])*b_inr,
                        +(omega_2 - omega_b - A[2]/2*b2)*vect[2]
                        - chim*p0*vect[0]
                        - kappas[2]/2*vect[3]
                        + np.sqrt(kappas[2])*b_ini])
                          
        return res
    sol = newton_krylov(optimizer, vect0/np.sqrt(kappas[1]))
#jac = jaco(p0, delta_1, delta_2, A[1], A[2], chim, kappas[1], kappas[2], a_inr, a_ini, b_inr, b_ini)).x
    a_stat = sol[0] + 1j * sol[1]
    return (-a_in + np.sqrt(kappas[1])*a_stat)/a_in



def gain_nv(s, phi_ext, a_in, b_in, p0, f1, fp):
    return np.abs(solve_stat(s, phi_ext, a_in, b_in, p0, f1, fp))**2

gain = np.vectorize(gain_nv)

def gain_dB(s, phi_ext, a_in, b_in, p0, f1, fp):
    return 10*np.log10(gain(s, phi_ext, a_in, b_in, p0, f1, fp))

def gain_res_dB(s, phi_ext, a_in, b_in, p0, f1):
    return 10*np.log10(gain_res(s, phi_ext, a_in, b_in, p0, f1))

def gain_f_dB(s, phi_ext, p0, f1, fp):
    return 10*np.log10(gain_f(s, phi_ext, p0, f1, fp))


def dBcp(s, phi_ext, p0, f1, fp):
    def optgain(a_in):
        #print(gain_dB(s, phi_ext, a_in, 0, p0, f1, fp))
        return -gain(s, phi_ext, np.abs(a_in), 0, p0, f1, fp)
    maximum = np.abs(fmin(optgain, a_in0, disp = 0)[0])
    ampl = (10*np.log10(-optgain(maximum))) >= 1.5 #checks that the signal is sufficiently amplified
    if ampl: 
        def f(a_in):
            return gain_dB(s, phi_ext, maximum, 0, p0, f1, fp) - gain_dB(s, phi_ext, a_in, 0, p0, f1, fp) - 1
    #print(2*gain_dB(s, phi_ext, 1e-3, 0, p0, f1, fp))
    #print(2*gain_dB(s, phi_ext, 1e-3, 0, p0, f1, fp) - 2*gain_dB(s, phi_ext, 1e-3, 0, p0, f1, fp) - 1)
    #print(2*gain_dB(s, phi_ext, 1e-3, 0, p0, f1, fp) - 2*gain_dB(s, phi_ext, 100*a_in0, 0, p0, f1, fp) - 1)
        return root_scalar(f, bracket = [maximum, 1e9*a_in0]).root
    else:
        return np.nan

def max_gain(s, phi_ext, p0, f1, fp):
    def optgain(a_in):
        #print(gain_dB(s, phi_ext, a_in, 0, p0, f1, fp))
        return -gain(s, phi_ext, np.abs(a_in), 0, p0, f1, fp)
    maximum = np.abs(fmin(optgain, a_in0)[0], disp = 0)
    return maximum




def gain_res(s, phi_ext, a_in, b_in, p0, f1):
    Np = np.abs(p0)**2
    freq, k, A, chi = s.f_k_A_chi(phi_ext)
    f_a = (freq[1] 
            - A[1] 
            - (chi[0, 1] + chi[2, 1] + chi[3, 1])/2
            - chi[3, 1]*Np)
    f_b = (freq[2] 
            - A[2] 
            - (chi[0, 2] + chi[1, 2] + chi[3, 2])/2
            - chi[3, 2]*Np)
    return gain(s, phi_ext, a_in, b_in, p0, f1, f_a+f_b)

def res_gain_bandwidth(s, phi_ext, a_in, b_in, p0, fp):
    a = s.eigenfrequencies(phi_ext)[1:3] + deltas(s, phi_ext, p0)

    def f(f1):
        return -gain(s, phi_ext, a_in, b_in, p0, f1, fp)
    sol = minimize(f, a[0])
    f_res = sol.x
    gain_res = -sol.fun
    print(f_res, gain_res)
    def g(f1):
        return gain(s, phi_ext, a_in, b_in, p0, f1, fp) - gain_res/2
    sol2 = root_scalar(g, bracket = [f_res, 2*f_res])
    sol1 = root_scalar(g, bracket = [f_res/2, f_res])
    bw = sol1.root, sol2.root
    delta = sol2.root - sol1.root
    
    return f_res, gain_res, bw, delta


def gain_f_res(s, phi_ext, p0, f1):
    Np = np.abs(p0)**2
    freq, k, A, chi = s.f_k_A_chi(phi_ext)
    f_a = (freq[1] 
            - A[1] 
            - (chi[0, 1] + chi[2, 1] + chi[3, 1])/2
            - chi[3, 1]*Np)
    f_b = (freq[2] 
            - A[2] 
            - (chi[0, 2] + chi[1, 2] + chi[3, 2])/2
            - chi[3, 2]*Np)
    return gain_f(s, phi_ext, p0, f1, f_a+f_b)

def deltas(s, phi_ext, p0): #resonance shift due to anharmonicities and cross-Kerr terms
    Np = np.abs(p0)**2
    freq, k, A, chi = s.f_k_A_chi(phi_ext)
    df_a = (- A[1] 
            - (chi[0, 1] + chi[2, 1] + chi[3, 1])/2
            - chi[3, 1]*Np)
    df_b = ( - A[2] 
            - (chi[0, 2] + chi[1, 2] + chi[3, 2])/2
            - chi[3, 2]*Np)
    return np.array([df_a, df_b])

def rho(s, phi_ext):
    kappas = s.coupling_rates(phi_ext)
    return 2*s.three_waves_term(phi_ext)/np.sqrt(kappas[1]*kappas[2])


def p0_max(s, phi_ext):
    kappas = s.coupling_rates(phi_ext)
    return 1/np.abs(2*s.three_waves_term(phi_ext)/np.sqrt(kappas[1]*kappas[2]))

def max_gain(s, phi_ext, a_in):
    def f(t):
        return -gain_nv(s, phi_ext, a_in, 0, t[1], t[0], t[2])
    a = s.eigenfrequencies(phi_ext)[1:3] + deltas(s, phi_ext, s.sweet_spot())
    return minimize(f, np.array([p0_max(s, s.sweet_spot())/10, a[0], a[0] + a[1]])).fun



def gain_exp(s, phi_ext, A_in, P_in, f1, fp):
    a_in = np.sqrt(a_in)
    p_in = np.sqrt(P_in)
    freq, kappas, A, chi = s.f_k_A_chi(phi_ext)
    omega_res = 2*pi*(freqs[3]-A[3])
    omega_p = 2*pi*fp
    p = np.sqrt(kappas[3]/((omega_p-omega_res)**2 + kappas[3]**2/4))*p_in
    return gain(s, phi_ext, a_in, p, f1, fp)

def gain_exp_dB(s, phi_ext, A_in, P_in, f1, fp):
    return 10*np.log10(gain_exp(s, phi_ext, A_in, P_in, f1, fp))

def P_in0_max(s, phi_ext): #for fp = sum of original resonances
    p0 = p0_max(s, phi_ext)
    freqs, kappas, A, chi = s.f_k_A_chi(phi_ext)
    omega_res = 2*pi*(freqs[3]-A[3])
    omega_p = 2*pi*(freqs[1]+ freqs[2])
    print(p0, omega_res, omega_p)
    print(1/kappas[3]*((omega_p-omega_res)**2 + kappas[3]**2/4))
    return p0**2/kappas[3]*((omega_p-omega_res)**2 + kappas[3]**2/4)

def P_in_max(s, phi_ext, fp): #for fp = sum of original resonances
    p0 = p0_max(s, phi_ext)
    freqs, kappas, A, chi = s.f_k_A_chi(phi_ext)
    omega_res = 2*pi*(freqs[3]-A[3])
    omega_p = 2*pi*fp
    print(p0, omega_res, omega_p)
    print(1/kappas[3]*((omega_p-omega_res)**2 + kappas[3]**2/4))
    return p0**2/kappas[3]*((omega_p-omega_res)**2 + kappas[3]**2/4)

def p(s, phi_ext, P_in):
    p_in = np.sqrt(P_in)
    freq, kappas, A, chi = s.f_k_A_chi(phi_ext)
    omega_res = 2*pi*(freqs[3]-A[3])
    omega_p = 2*pi*fp
    p = np.sqrt(kappas[3]/((omega_p-omega_res)**2 + kappas[3]**2/4))*p_in
    return p
    


def I_snail(s, mode, p, phi_ext):
    #returns the phasor I for one snail in converted units
    phi0 = 0.5
    return 4*pi**2*s.phis_zpf(phi_ext)[mode]*2*(s.coeffs(phi_ext)[0])/phi0*(2*p)

def I_junctions(s, mode, p, phi_ext):
    #returns the phasors for the two sides of the snail
    return I_snail(s, mode, p, phi_ext)/(1+s.alpha/3), I_snail(s, mode, p, phi_ext)/(1+3/s.alpha) #3 junctions, alpha junction













######## experimental
def expr(): #gives the expression of the jacobian of the optimizer function
    vect = [sp.symbols('ar'), sp.symbols('ai'), sp.symbols('br'), sp.symbols('bi')]

    p0 = sp.symbols('p0')
    delta_1 = sp.symbols('delta_1')
    delta_2 = sp.symbols('delta_2')
    A_1 = sp.symbols('A_1')
    A_2 = sp.symbols('A_2')
    chi_m = sp.symbols('chi_m')
    kappa_1 = sp.symbols('kappa_1')
    kappa_2 = sp.symbols('kappa_2')
    a_inr = sp.symbols('a_inr')
    a_ini = sp.symbols('a_ini')
    b_inr = sp.symbols('b_inr')
    b_ini = sp.symbols('b_ini')

    vects = sp.Matrix(vect)

    exp = sp.Matrix([-(delta_1 - A_1/2*(vect[0]**2 + vect[1]**2))*vect[1]
                - chi_m*p0*vect[3]
                - kappa_1/2*vect[0]
                + sp.sqrt(kappa_1)*a_inr,
                +(delta_1 - A_1/2*(vect[0]**2 + vect[1]**2))*vect[0]
                - chi_m*p0*vect[2]
                - kappa_1/2*vect[1]
                + sp.sqrt(kappa_1)*a_ini,
                -(delta_2 - A_2/2*(vect[2]**2 + vect[3]**2))*vect[3]
                - chi_m*p0*vect[1]
                - kappa_2/2*vect[2]
                + sp.sqrt(kappa_2)*b_inr,
                +(delta_2 - A_2/2*(vect[2]**2 + vect[3]**2))*vect[2]
                - chi_m*p0*vect[0]
                - kappa_2/2*vect[3]
                + sp.sqrt(kappa_2)*b_ini])

    jac = lambdify(vect + [p0, delta_1, delta_2, A_1, A_2, chi_m, kappa_1, kappa_2, a_inr, a_ini, b_inr, b_ini], 
               exp.jacobian(vects))
    return jac

jac = expr()

def jaco(p0, delta_1, delta_2, A_1, A_2, chi_m, kappa_1, kappa_2, a_inr, a_ini, b_inr, b_ini):
    #computes the jacobian of the optimizer function
    @jit
    def f(vect):
        return jac(vect[0], vect[1], vect[2], vect[3], 
                   p0, delta_1, delta_2, A_1, A_2, chi_m, 
                   kappa_1, kappa_2, a_inr, a_ini, b_inr, b_ini)
    return f

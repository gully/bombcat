#!/Users/gully/anaconda/bin/python
# Filename: etalon.py

import numpy as np

@np.vectorize
def sellmeier_Si(lam_nm):
    ''' return the Si refractive index, n, for a given wavelength
        the default temperature is 295.0 K
        Sellmeier coeffs are from Frey et al. 2006 (NASA CHARMS group)

        Relationship is valid are valid over the range:
        20<T<300
        1.1< wl <5.6
        lam can be a vector or scalar
        
    '''
    t_k = 295.0
    
    lam_um = lam_nm/1000.0

    if (lam_um < 1.0) or (lam_um > 5.6):
        raise Exception
    #if (t_K < 20) or (t_K > 400):
    #    raise Exception

    s1j = [10.4907,-2.08020E-04,4.21694E-06,-5.82298E-09,3.44688E-12]
    s2j = [-1346.61,29.1664,-0.278724,1.05939E-03,-1.35089E-06]
    s3j = [4.42827E+07,-1.76213E+06,-7.61575E+04,678.414,103.243]

    s1 = np.polynomial.polynomial.polyval(t_k, s1j)
    s2 = np.polynomial.polynomial.polyval(t_k, s2j)
    s3 = np.polynomial.polynomial.polyval(t_k, s3j)

    l1j = [0.299713,-1.14234E-05,1.67134E-07,-2.51049E-10,2.32484E-14]
    l2j = [-3.51710E+03,42.3892,-0.357957,1.17504E-03,-1.13212E-06]
    l3j = [1.71400E+06,-1.44984E+05,-6.90744E+03,-39.3699,23.5770]

    l1 = np.polynomial.polynomial.polyval(t_k, l1j)
    l2 = np.polynomial.polynomial.polyval(t_k, l2j)
    l3 = np.polynomial.polynomial.polyval(t_k, l3j)

    l_2 = lam_um**2
    n2 = 1.0 + (s1*l_2/(l_2-l1**2)) + (s2*l_2/(l_2-l2**2)) + (s3*l_2/(l_2-l3**2))
    n=np.sqrt(n2)

    return n

@np.vectorize
def T_gap_Si(lam_nm, dgap_nm):
    ''' return the Transmission spectrum for a given axial extent of Air gap in Si
        Transmission is absolute
    ''' 
    
    # Determine the refractive index
    n1 = sellmeier_Si(lam_nm)

    #Silicon reflectance (Fresnel losses at 1 interface)
    R0 = ((n1-1.0)/(n1+1.0))**2.0

    #Coefficient of Finesse
    F = 4.0*R0/(1.0-R0)**2.0

    delta = 2.0*3.141592654*dgap_nm/lam_nm

    T_net=2.0*n1/(1.0+2.0*n1*F*np.sin(delta)**2.0+n1**2.0)
    T_old=1.0/(1.0+F*np.sin(delta)**2.0)

    return T_net


def T_gap_Si_fast(lam_nm, dgap_nm, n1):
    ''' return the Transmission spectrum for a given axial extent of Air gap in Si
    	This fast version requires input of the refractive index, n(lam)
        Transmission is absolute
    ''' 

    #Silicon reflectance (Fresnel losses at 1 interface)
    R0 = ((n1-1.0)/(n1+1.0))**2.0

    #Coefficient of Finesse
    F = 4.0*R0/(1.0-R0)**2.0

    delta = 2.0*3.141592654*dgap_nm/lam_nm

    T_net=2.0*n1/(1.0+2.0*n1*F*np.sin(delta)**2.0+n1**2.0)

    return T_net



def T_gap_Si_withFF(wl_nm, d_gap, ff):
    '''computes weighted average model for a given gap size and fill factor
        
        *inputs*
        wl: wavelength in nm
        d_gap: axial extent of the gap in nm
        ff: fill factor of gap area as a fraction (0<ff<1)
    '''
    return ff*T_gap_Si(wl_nm, d_gap) + (1.0-ff)*T_gap_Si(wl_nm, 0.0)
    
    
def T_gap_Si_withFF_fast(wl_nm, d_gap, ff, n1):
    '''computes weighted average model for a given gap size and fill factor
        
        *inputs*
        wl: wavelength in nm
        d_gap: axial extent of the gap in nm
        ff: fill factor of gap area as a fraction (0<ff<1)
    '''
    return ff*T_gap_Si_fast(wl_nm, d_gap, n1) + (1.0-ff)*T_gap_Si_fast(wl_nm, 0.0, n1)
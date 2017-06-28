# convert material parameters of Palasantzas to frequencies 
# which can be used in a Lorentz osciallator model
from math import sqrt

def transform(params):
    """ take a list of tuples containing to values
        which correspond to Palasantzas osciallator strength
        and the osciallator frequency in units of electronvolts
    """
    eV = 1.60218e-19
    hbar = 1.05457e-34
    omega_eV = eV/hbar
    t_params = params.copy()
    for i, osciallator in enumerate(params):
        t_params[i][0] = sqrt(osciallator[0]*(osciallator[1]*omega_eV)**2)
        t_params[i][1] = osciallator[1]*omega_eV
    return t_params

liste =    [[7.84e-1, 4.11e-2],
            [2.03e-1, 1.12e-1],
            [4.17e-1, 1.12e-1],
            [3.93e-1, 1.11e-1],
            [5.01e-2, 1.45e+1],
            [8.20e-1, 1.70e+1],
            [2.17e-1, 8.14e+0],
            [5.50e-2, 9.16e+1]]

erg = transform(liste)

for item in erg:
    print("wp = %.2e, w1 = %.2e" %(item[0], item[1]))

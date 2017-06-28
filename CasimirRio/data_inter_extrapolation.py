"""data_inter_extrapolation.py allows to use measured optical
   dielectric data together with a low-frequency Drude 
   interpolation to obtain broadband dielectric functions 
   necessary in Casimir calculations.

"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


class data_interp_extrap:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def interpolate(self, x_new):
        """interpolate y-data where possible and call extrapolation routine
           with given extrapolation function for values of x_new outside of
           the interpolation region

        """
        x_l = x_new[x_new > self.xdata[0]][0]
        x_u = x_new[x_new < self.xdata[-1]][-1]
        x_ip = x_new[np.where((x_new >= x_l) & (x_new <= x_u))]
        interpolator = interp1d(self.xdata, self.ydata)
        return [interpolator(x_ip), (x_l, x_u)]

    def epsilon_drude(self, omega, omega_p, gamma):
        return 1 + omega_p**2/(omega*gamma+omega**2)

    def fit(self, fitfunc, omega_desired):
        """take data on a grid and give back data on a desired grid by using
           the fitfunc fitting function

        """
        fit_params, opt = curve_fit(fitfunc, self.xdata, self.ydata)
        return fitfunc(omega_desired, fit_params[0], fit_params[1])

    def get_drude_data(self, omega):
        if type(omega) == float:
            omega = np.array([omega])
        fitfunc = self.epsilon_drude
        if len(omega) == 1:
            omega_p_gold = 1.4e16/(3e14)
            gamma_gold = 5.3e13/(3e14)
            return self.epsilon_drude(omega, omega_p_gold, gamma_gold)
        data_2, bounds = self.interpolate(omega)
        x_1 = omega[omega < bounds[0]]
        x_3 = omega[omega > bounds[1]]
        data_1 = self.fit(fitfunc, x_1)
        data_3 = self.fit(fitfunc, x_3)
        result = np.append(np.append(data_1, data_2, axis=0), data_3, axis=0)
        return result

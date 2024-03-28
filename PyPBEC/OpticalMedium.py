import numpy as np
from scipy import constants as sc
from scipy.interpolate import interp1d
from pathlib import Path
from scipy.special import erf as Erf
import pandas as pd
import sys
import os
import csv
from scipy.integrate import quad

class OpticalMedium():

	available_media = list()
	available_media.append("Rhodamine6G")
	available_media.append("InGaAs_QW")

	def __init__(self, optical_medium):

		"""
			Initiazies an optical medium object.

			Parameters:

				optical_medium (str):		Optical medium

		"""

		if not type(optical_medium) == str:
			raise Exception("optical_medium is expected to be a string")

		if not optical_medium in self.available_media:
			raise Exception(optical_medium+" is an unknown optical medium")

		if optical_medium == "Rhodamine6G":
			self.medium = Rhodamine6G()

		if optical_medium == "InGaAs_QW":
			self.medium = InGaAs_QW()



	def get_rates(self, lambdas, **kwargs):

		"""
			Calculates the rates of absorption and emission, for a specific optical medium.

			Parameters:

				lambdas (list, or other iterable): Wavelength points where the rates are to be calculated. Wavelength is in meters
				other medium specific arguments
				
		"""

		return self.medium.get_rates(lambdas=lambdas, **kwargs)




class Rhodamine6G(OpticalMedium):

	def __init__(self):
		pass


	def get_rates(self, lambdas, dye_concentration, n):

		"""
			Rates for Rhodamine 6G

			Parameters:

				lambdas (list, or other iterable):  	Wavelength points where the rates are to be calculated. Wavelength is in meters
				dye_concentration (float):				In mM (milimolar) 1 mM = 1 mol / m^3			
				n (float): 								index of refraction

		"""

		# absorption data
		min_wavelength = 480
		max_wavelength = 650
		absorption_spectrum_datafile = Path("data") / 'absorption_cross_sections_R6G_in_EthyleneGlycol_corrected.csv'
		absorption_spectrum_datafile = Path(os.path.dirname(os.path.abspath(__file__))) / absorption_spectrum_datafile
		raw_data2 = pd.read_csv(absorption_spectrum_datafile)
		initial_index = raw_data2.iloc[(raw_data2['wavelength (nm)']-min_wavelength).abs().argsort()].index[0]
		raw_data2 = raw_data2.iloc[initial_index:].reset_index(drop=True)
		final_index = raw_data2.iloc[(raw_data2['wavelength (nm)']-max_wavelength).abs().argsort()].index[0]
		raw_data2 = raw_data2.iloc[:final_index].reset_index(drop=True)
		absorption_data = raw_data2
		absorption_data_normalized = absorption_data['absorption cross-section (m^2)'].values / np.max(absorption_data['absorption cross-section (m^2)'].values)
		absorption_spectrum = np.squeeze(np.array([[absorption_data['wavelength (nm)'].values], [absorption_data_normalized]], dtype=float))
		interpolated_absorption_spectrum = interp1d(absorption_spectrum[0,:], absorption_spectrum[1,:], kind='cubic')
		
		# emission data
		fluorescence_spectrum_datafile =  Path("data") / 'fluorescence_spectrum_R6G_in_EthyleneGlycol_corrected.csv'
		fluorescence_spectrum_datafile = Path(os.path.dirname(os.path.abspath(__file__))) / fluorescence_spectrum_datafile
		raw_data = pd.read_csv(fluorescence_spectrum_datafile)
		initial_index = raw_data.iloc[(raw_data['wavelength (nm)']-min_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[initial_index:].reset_index(drop=True)
		final_index = raw_data.iloc[(raw_data['wavelength (nm)']-max_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[:final_index].reset_index(drop=True)
		fluorescence_data = raw_data
		fluorescence_data_normalized = fluorescence_data['fluorescence (arb. units)'].values / np.max(fluorescence_data['fluorescence (arb. units)'].values)
		emission_spectrum = np.squeeze(np.array([[fluorescence_data['wavelength (nm)'].values], [fluorescence_data_normalized]], dtype=float))
		interpolated_emission_spectrum = interp1d(emission_spectrum[0,:], emission_spectrum[1,:], kind='cubic')

		# Uses both datasets
		if np.min(1e9*np.array(lambdas)) < 480 or np.max(1e9*np.array(lambdas)) > 650:
			raise Exception('*** Restrict wavelength to the range between 480 and 650 nm ***')

		temperature = 300
		lamZPL = 545e-9
		n_mol_per_vol= dye_concentration*sc.Avogadro
		peak_Xsectn = 2.45e-20*n_mol_per_vol*sc.c/n
		wpzl = 2*np.pi*sc.c/lamZPL/1e12

		def freq(wl):
			return 2*np.pi*sc.c/wl/1e12
		def single_exp_func(det):
			f_p = 2*np.pi*sc.c/(wpzl+det)*1e-3
			f_m = 2*np.pi*sc.c/(wpzl-det)*1e-3
			return (0.5*interpolated_absorption_spectrum(f_p)) + (0.5*interpolated_emission_spectrum(f_m))
		def Err(det):
			return Erf(det*1e12)
		def single_adjust_func(det):
			return ((1+Err(det))/2.0*single_exp_func(det)) + ((1-Err(det))/2.0*single_exp_func(-1.0*det)*np.exp(sc.h/(2*np.pi*sc.k*temperature)*det*1e12)) 
			
		emission_rates = np.array([single_adjust_func(-1.0*freq(a_l)+wpzl) for a_l in lambdas])*peak_Xsectn
		absorption_rates = np.array([single_adjust_func(freq(a_l)-wpzl) for a_l in lambdas])*peak_Xsectn

		return absorption_rates, emission_rates


class InGaAs_QW(OpticalMedium):

	def __init__(self):
		pass

	def get_rates(self, lambdas, mode):
		"""
			Rates for InGaAs single quantum well. Converts from absorption percentage chance to a rate using average time taken to travel cavity.

			Parameters:

				lambdas (list, or other iterable):  	Wavelength points where the rates are to be calculated. Wavelength is in meters
				n (float): 								index of refraction

		"""

		# absorption data
		min_wavelength = 900
		max_wavelength = 970
		absorption_spectrum_datafile = Path("data") / 'qw_absorption.csv'
		absorption_spectrum_datafile = Path(os.path.dirname(os.path.abspath(__file__))) / absorption_spectrum_datafile
		raw_data2 = pd.read_csv(absorption_spectrum_datafile)
		initial_index = raw_data2.iloc[(raw_data2['wavelength (nm)'] - min_wavelength).abs().argsort()].index[0]
		raw_data2 = raw_data2.iloc[initial_index:].reset_index(drop=True)
		final_index = raw_data2.iloc[(raw_data2['wavelength (nm)'] - max_wavelength).abs().argsort()].index[0]
		raw_data2 = raw_data2.iloc[:final_index].reset_index(drop=True)
		absorption_data = raw_data2
		absorption_data_normalized = absorption_data['absorption per pass (percentage)'].values / np.max(
			absorption_data['absorption per pass (percentage)'].values)
		absorption_spectrum = np.squeeze(
			np.array([[absorption_data['wavelength (nm)'].values], [absorption_data_normalized]], dtype=float))
		interpolated_absorption_spectrum = interp1d(absorption_spectrum[0, :], absorption_spectrum[1, :], kind='cubic')

		# emission data
		fluorescence_spectrum_datafile = Path("data") / 'qw_emission.csv'
		fluorescence_spectrum_datafile = Path(
			os.path.dirname(os.path.abspath(__file__))) / fluorescence_spectrum_datafile
		raw_data = pd.read_csv(fluorescence_spectrum_datafile)
		initial_index = raw_data.iloc[(raw_data['wavelength (nm)'] - min_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[initial_index:].reset_index(drop=True)
		final_index = raw_data.iloc[(raw_data['wavelength (nm)'] - max_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[:final_index].reset_index(drop=True)
		fluorescence_data = raw_data
		fluorescence_data_normalized = fluorescence_data['fluorescence (arb. units)'].values / np.max(
			fluorescence_data['fluorescence (arb. units)'].values)
		emission_spectrum = np.squeeze(
			np.array([[fluorescence_data['wavelength (nm)'].values], [fluorescence_data_normalized]], dtype=float))
		interpolated_emission_spectrum = interp1d(emission_spectrum[0, :], emission_spectrum[1, :], kind='cubic')

		# Uses both datasets
		if np.min(1e9 * np.array(lambdas)) < 900 or np.max(1e9 * np.array(lambdas)) > 969:
			raise Exception('*** Restrict wavelength to the range between 900 and 969 nm ***')

		temperature = 300

		#Create function to calculate absorption rates
		def absorption_rates_func(lamb, mode):
			cavity_length = lamb * mode / 2
			abs_time = 2*cavity_length/(sc.c*interpolated_absorption_spectrum(lamb))
			absorption_rate = 1/abs_time
			return absorption_rate

		#Normalise emission rates to absorption
		abs_sum, _ = quad(absorption_rates_func, 900, 969, args=(mode,))
		emi_sum, _ = quad(interpolated_emission_spectrum, 900, 969)
		sf = abs_sum/emi_sum

		emission_rates = interpolated_emission_spectrum(lambdas*1e9)*sf
		absorption_rates = absorption_rates_func(lambdas*1e9, mode)



		return absorption_rates, emission_rates
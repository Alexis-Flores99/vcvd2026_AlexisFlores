"""
This script simulates and plots tyre forces.
Author: Alexis Flores Avina
Date: 01/11/2026
"""
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import g

from s2610787012 import G_CAMBER  # Gravitational constant

parser = argparse.ArgumentParser(description="Simulate and  plot tyre forces.")
parser.add_argument("--weight", type=float, default=1500, help="weight in kg")
parser.add_argument("--slip_angle", type=int, default=2, help="List of slip angles in degrees")
parser.add_argument("--friction", type=float, default=0.2, help="Friction coefficient")
parser.add_argument("--output_file", type=str, default="Tyre_Forces.png", help="Path to save the plot")
args = parser.parse_args()

def brake_force(long_slip, f_z):
    """
    Implements the brake_force Formula.
    :param long_slip: Longitudinal slip (%).
    :param B: Stiffness factor.
    :param C: Shape factor.
    :param D: Peak factor.
    :param E: Curvature factor.
    :param Fz: Vertical load, equal load distribution over 4 wheels
    :return: Brake Force value.
    APPENDIX, pag. 2.204, Egbert Bakker, Lars Nyborg and Hans B. Pacejka,
    "Tyre Modelling for Use in Vehicle Dynamics Studies", SAE Transactions ,
    1987, Vol. 96, Section 2 (1987),
    pp. 190-204, 1987
    """
    c_fx = 1.65
    d_fx = ((a_1_fx) * f_z ** 2) + ((a_2_fx) * f_z)
    b_fx = (((a_3_fx) * f_z ** 2) + (a_4_fx) * f_z) / (c_fx * d_fx * math.exp((a_5_fx * f_z)))
    e_fx = (a_6_fx) * f_z ** 2 + (a_7_fx) * f_z + a_8_fx

    phi = (1 - e_fx) * long_slip + (e_fx / b_fx) * np.arctan(b_fx * long_slip) * 180 / np.pi
    return d_fx * np.sin(c_fx * np.arctan(b_fx * phi))

def side_force(f_z):
    """
    Implements the side_force Formula.
    :param alpha: slip angle.
    :param B: Stiffness factor.
    :param C: Shape factor.
    :param D: Peak factor.
    :param E: Curvature factor.
    :param Fz: Vertical load, equal load distribution over 4 wheels
    :return: Side Force value.
    pag. 2.204, Egbert Bakker, Lars Nyborg and Hans B. Pacejka,
    "Tyre Modelling for Use in Vehicle Dynamics Studies", SAE Transactions ,
    1987, Vol. 96, Section 2 (1987), pp. 190-204, 1987
    """
    c_fy = 1.3
    d_fy = ((a_1_fy) * f_z ** 2) + ((a_2_fy) * f_z)
    b_fy = (((a_3_fy) * f_z ** 2) + (a_4_fy) * f_z) / (c_fy * d_fy * math.exp((a_5_fy * f_z)))
    e_fy = (a_6_fy) * f_z ** 2 + (a_7_fy) * f_z + a_8_fy
    d_s_h = a_9_fy * G_CAMBER
    d_s_v = (a_10_fy *f_z**2 + a_11_fy * f_z) * G_CAMBER

    phi = (1 - e_fy) * args.slip_angle + (e_fy / b_fy) * np.arctan(b_fy * args.slip_angle) * 180 / np.pi
    return d_fy * np.sin(c_fy * np.arctan(b_fy * phi)) + d_s_h + d_s_v

def calculate_fx(k, fz):
    """
    Calculate longitudinal and lateral forces using the theoretical slip.
    :param k: Longitudinal slip (%).
    :param alpha: Slip angle (degrees).
    :param B,C,D,E: Tyre model coefficients.
    :return: Longitudinal force (Fx), lateral force (Fy).
    Eq (23) pag 2.201, Egbert Bakker, Lars Nyborg and Hans B. Pacejka,
    "Tyre Modelling for Use in Vehicle Dynamics Studies", SAE Transactions , 1987,
    Vol. 96, Section 2 (1987), pp. 190-204, 1987     
    """
    sigma_x = -k / (1 + k)
    sigma_y = np.tan(alpha*np.pi/180) / (1 + k)
    sigma = np.sqrt(sigma_x**2 + sigma_y**2)
    f_x0 = brake_force(k, fz)
    fx = -(sigma_x / sigma)*mu * f_x0
    return fx

def calculate_fy(k,fz):
    """
    Calculate longitudinal and lateral forces using the theorical slip.
    :param k: Longitudinal slip (%).
    :param alpha: Slip angle (degrees).
    :param B, C, D, E: Tyre model coefficients.
    :return: Longitudinal force (Fx), lateral force (Fy).
    Eq (24) pag. 2.201, Egbert Bakker, Lars Nyborg and Hans B. Pacejka, 
    "Tyre Modelling for Use in Vehicle Dynamics Studies", SAE Transactions , 1987, 
    Vol. 96, Section 2 (1987), pp. 190-204, 1987
    """
    sigma_x = -k / (1 + k)
    sigma_y = -np.tan(alpha*np.pi/180) / (1 + k)
    sigma = np.sqrt(sigma_x**2 + sigma_y**2)
    f_y0 = side_force(fz)
    fy = -(sigma_y/sigma)*mu*f_y0
    return fy

# Load tyre coefficients from CSV file
coefficients = pd.read_csv(r".\tyre_coefficients.csv")

# Vehicle and tyre parameters
load = (args.vehicle_mass * g)/4000
mu = args.friction
G_CAMBER = 0

# Simulation parameters
kappa = np.linspace(0, 1, 1000)  # Longitudinal slip from 0% to 100%
alpha = args.slip_angle  # slip angles in degrees

# Extract coefficients for tyre formula from the CSV file
# Table 2 & 3, pag. 2.195, Egbert Bakker, Lars Nyborg and Hans B. Pacejka,
# "Tyre Modelling for Use in Vehicle Dynamics Studies", SAE Transactions , 1987,
# Vol. 96, Section 2 (1987), pp. 190-204, 1987
a_1_fy = coefficients["a1"].iloc[0]
a_2_fy = coefficients["a2"].iloc[0]
a_3_fy = coefficients["a3"].iloc[0]
a_4_fy = coefficients["a4"].iloc[0]
a_5_fy = coefficients["a5"].iloc[0]
a_6_fy = coefficients["a6"].iloc[0]
a_7_fy = coefficients["a7"].iloc[0]
a_8_fy = coefficients["a8"].iloc[0]
a_9_fy = coefficients["a9"].iloc[0]
a_10_fy = coefficients["a10"].iloc[0]
a_11_fy = coefficients["a11"].iloc[0]
a_12_fy = coefficients["a12"].iloc[0]
a_13_fy = coefficients["a13"].iloc[0]

a_1_fx = coefficients["a1"].iloc[2]
a_2_fx = coefficients["a2"].iloc[2]
a_3_fx = coefficients["a3"].iloc[2]
a_4_fx = coefficients["a4"].iloc[2]
a_5_fx = coefficients["a5"].iloc[2]
a_6_fx = coefficients["a6"].iloc[2]
a_7_fx = coefficients["a7"].iloc[2]
a_8_fx = coefficients["a8"].iloc[2]

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(kappa*100,calculate_fy(kappa,load),\
         label="Fy - "+str(((load*1000)))+" N")
plt.plot(kappa*100,calculate_fx(kappa,load),\
         label="Fx - "+str(((load*1000)))+" N")

plt.title("Side and Brake Forces vs Longitudinal Slip"+"\n"+ \
           " Slip Angle =" + str(alpha)+"°"\
              + "   Friction Coeficient =" + str(mu) )
plt.xlabel("Longitudinal Slip (%)")
plt.ylabel("Force (N)")
plt.legend(loc="best")
plt.grid(True)

plt.savefig(str(args.output_file))

plt.show()

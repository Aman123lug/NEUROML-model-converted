import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
dt = 0.01  # time step size (ms)
T = 50  # simulation time (ms)

# Define the Hodgkin-Huxley neuron model parameters
C_m = 1.0  # membrane capacitance (uF/cm^2)
g_Na = 120.0  # maximum sodium conductance (mS/cm^2)
g_K = 36.0  # maximum potassium conductance (mS/cm^2)
g_L = 0.3  # leak conductance (mS/cm^2)
E_Na = 50.0  # sodium reversal potential (mV)
E_K = -77.0  # potassium reversal potential (mV)
E_L = -54.387  # leak reversal potential (mV)

# Define the initial conditions
V_m = -65.0  # initial membrane potential (mV)
m = 0.0529  # initial sodium activation gating variable
h = 0.5961  # initial sodium inactivation gating variable
n = 0.3177  # initial potassium activation gating variable

# Define the input current
I = np.zeros(int(T/dt))
I[1000:2000] = 5  # input current from 1s to 2s

# Define the membrane currents
def sodium_current(V, m, h):
    return g_Na * m**3 * h * (V - E_Na)

def potassium_current(V, n):
    return g_K * n**4 * (V - E_K)

def leak_current(V):
    return g_L * (V - E_L)

# Define the gating variable dynamics
def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))

def beta_m(V):
    return 4 * np.exp(-0.0556 * (V + 65))

def alpha_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65))

def beta_h(V):
    return 1 / (1 + np.exp(-0.1 * (V + 35)))

def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55)))

def beta_n(V):
    return 0.125 * np.exp(-0.0125 * (V + 65))

# Simulate the neuron
V = np.zeros(int(T/dt))
m_vec = np.zeros(int(T/dt))
h_vec = np.zeros(int(T/dt))
n_vec = np.zeros(int(T/dt))
V[0] = V_m
m_vec[0] = m
h_vec[0] = h
n_vec[0] = n

for i in range(1, int(T/dt)):
    # Compute the membrane currents
    I_Na = sodium_current(V[i-1], m, h)
    I_K = potassium_current(V[i-1], n)
    I_L = leak_current(V[i-1])
    I_ion = I[i-1] - (I_Na + I_K + I_L)

    # Update the gating variables using the Euler method
    m += dt * (alpha_m(V[i-1]) * (1 - m) - beta_m(V[i-1]) * m)
   

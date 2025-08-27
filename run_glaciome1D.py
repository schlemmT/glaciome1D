# try setting dH/dt and dL/dt = 0 to speed up solver?


import numpy as np

import os

# from glaciome1D_dimensional import glaciome, basic_figure, plot_basic_figure, constants
from glaciome1D import glaciome, basic_figure, plot_basic_figure, constants



import pickle

import time

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: 
    Jason Amundson
    University of Alaska Southeast
    jmamundson@alaska.edu
"""


# basic parameters needed for setting up the model; later will modify this so that 
# the fjord geometry can be passed through
constant = constants()

n_pts = 21 # number of grid points
L = 1e4 # ice melange length
Ut = 0.6e4 # glacier terminus velocity [m/a]; treated as a constant
Uc = 0.6e4 # glacier calving rate [m/a]; treated as a constant
Ht = 600 # terminus thickness
n = 101 # number of time steps
dt = 0.01 # 1/(n_pts-1)/10 # time step [a]; needs to be quite small for this to work
B = -219.15 # initial melt rate, spatially uniform [m a^{-1}]; you can change this to vary spatially after initializing the class

# DEBUG: Print initial parameters
print("=== INITIAL PARAMETERS ===")
print(f"Grid points: {n_pts}")
print(f"Length: {L} m")
print(f"Terminus velocity: {Ut} m/a")
print(f"Calving rate: {Uc} m/a")
print(f"Terminus thickness: {Ht} m")
print(f"Time steps: {n}")
print(f"Time step: {dt} a")
print(f"Melt rate: {B} m/a")
print("=========================")

# specifying fjord geometry
X_fjord = np.linspace(-200e3,200e3,101)
Wt = 4000
W_fjord = Wt + 0/10000*X_fjord

print("=== FJORD GEOMETRY SETUP ===")
print(f"Fjord width at terminus: {Wt} m")
print(f"Fjord width variation: {0/10000} (currently set to 0)")
print(f"Fjord coordinates range: [{X_fjord.min():.0f}, {X_fjord.max():.0f}] m")
print("===========================")


# set up basic figure
print("=== SETTING UP FIGURE ===")
axes, color_id = basic_figure(n, dt)
print("Figure setup completed")
print("=========================")

print("=== INITIALIZING GLACIOME MODEL ===")
data = glaciome(n_pts, dt, L, Ut, Uc, Ht, B, X_fjord, W_fjord)
print("Glaciome model initialized successfully")
print("=========================")

# DEBUG: Check model initialization
print("=== MODEL INITIALIZATION ===")
print(f"Model length: {data.L} m")
print(f"Model time: {data.t} years")
print(f"Velocity range: [{data.U.min():.2f}, {data.U.max():.2f}] m/a")
print(f"Thickness range: [{data.H.min():.2f}, {data.H.max():.2f}] m")
print(f"Width range: [{data.W.min():.2f}, {data.W.max():.2f}] m")
print("===========================")

start = time.time()
print(f"=== TIMING STARTED: {time.strftime('%H:%M:%S')} ===")

# DEBUG: Set breakpoint here for diagnostic solve
print("=== STARTING DIAGNOSTIC SOLVE ===")
try:
    print("=== DIAGNOSTIC SOLVE - CHECKING INITIAL STATE ===")
    print(f"Initial U range: [{data.U.min():.6f}, {data.U.max():.6f}]")
    print(f"Initial gg range: [{data.gg.min():.6f}, {data.gg.max():.6f}]")
    print(f"Initial muW range: [{data.muW.min():.6f}, {data.muW.max():.6f}]")
    
    # Set breakpoint here to inspect before solver
    # breakpoint()  # DEBUG: Pause here to inspect initial state
    
    print("Calling diagnostic solver with Levenberg-Marquardt method...")
    data.diagnostic()
    print("=== DIAGNOSTIC SOLVE COMPLETED SUCCESSFULLY ===")
except Exception as e:
    print(f"DIAGNOSTIC SOLVE FAILED: {e}")
    print("Trying with hybrid solver...")
    try:
        print("Calling diagnostic solver with hybrid method...")
        data.diagnostic(method='hybr')
        print("=== DIAGNOSTIC SOLVE COMPLETED WITH HYBRID SOLVER ===")
    except Exception as e2:
        print(f"HYBRID SOLVER ALSO FAILED: {e2}")
        raise

print("=== PLOTTING INITIAL STATE ===")
plot_basic_figure(data, axes, color_id, 0)
print("Initial state plotted")

print("=== SAVING INITIAL DATA ===")
data.save('melange_initial_state.pickle')
print("Initial data saved to melange_initial_state.pickle")

print("=== ADJUSTING TIME STEP ===")
print(f"Changing dt from {data.dt} to 0.1")
data.dt = 0.1
print("Time step adjusted")
# data.steadystate()

# j = 1
# while j<50:
#     print(j)
#     data.prognostic(method='hybr')
#     plot_basic_figure(data, axes, color_id, 50)
#     j+=1

# DEBUG: Set breakpoint here for steady state solve
print("=== STARTING STEADY STATE SOLVE ===")
print("Calling steadystate solver with hybrid method...")
data.steadystate(method='hybr')
print("=== STEADY STATE SOLVE COMPLETED ===")

print("=== PLOTTING STEADY STATE ===")
plot_basic_figure(data, axes, color_id, 100)
print("Steady state plotted")

print("=== SAVING STEADY STATE DATA ===")
data.save('melange_steady_state.pickle')
print("Steady state data saved to melange_steady_state.pickle")

stop = time.time()
print(f"=== TIMING COMPLETED: {time.strftime('%H:%M:%S')} ===")
print(f"Total execution time: {stop-start:.2f} seconds")

#%%
print("=== STARTING ADDITIONAL DIAGNOSTIC SOLVE ===")
data.diagnostic()
print("Additional diagnostic solve completed")

print("=== PLOTTING ADDITIONAL STATE ===")
plot_basic_figure(data, axes, color_id, 0)
print("Additional state plotted")

print("=== STARTING PROGNOSTIC SIMULATION LOOP ===")
print("Will run up to 49 prognostic steps...")

for j in np.arange(1,50):
    print(f"\n=== PROGNOSTIC STEP {j}/49 ===")
    print(f"Step start time: {time.strftime('%H:%M:%S')}")
    
    # Store previous state for comparison
    old_L = data.L
    old_volume = np.trapz(data.H, data.X_)*data.W[0]/1e9
    print(f"Previous length: {old_L:.2f} m")
    print(f"Previous volume: {old_volume:.4f} km³")
    
    try:
        print("Calling prognostic solver...")
        data.prognostic()
        print("Prognostic solve completed successfully")
        
        # Check for physical changes
        dL = data.L - old_L
        current_volume = np.trapz(data.H, data.X_)*data.W[0]/1e9
        dV = current_volume - old_volume
        
        print(f"Length change: {dL:.2f} m")
        print(f"Volume change: {dV:.4f} km³")
        print(f"Current length: {data.L:.2f} m")
        print(f"Current volume: {current_volume:.4f} km³")
        print(f"Current time: {data.t:.3f} years")
        
        # Check for instability
        if abs(dL) > 1000:
            print("WARNING: Large length change detected!")
            # breakpoint()  # Pause here to investigate
            
    except Exception as e:
        print(f"PROGNOSTIC STEP {j} FAILED: {e}")
        # breakpoint()  # Pause here to investigate
        break
    
    print("Plotting current state...")
    plot_basic_figure(data, axes, color_id, j)
    
    # Save data every 10 steps to avoid too many files
    if j % 10 == 0:
        print(f"Saving data at step {j}...")
        data.save(f'melange_step_{j:03d}.pickle')
        print(f"Data saved to melange_step_{j:03d}.pickle")
    
    print(f"Step {j} completed at {time.strftime('%H:%M:%S')}")

print("=== PROGNOSTIC SIMULATION LOOP COMPLETED ===")



#%%
print("=== STARTING SECOND STEADY STATE SOLVE ===")
start = time.time()
print(f"Second steady state solve started at: {time.strftime('%H:%M:%S')}")
data.steadystate()
stop = time.time()
print(f"Second steady state solve completed at: {time.strftime('%H:%M:%S')}")
print(f"Second steady state solve time: {stop-start:.2f} seconds")

print("=== SETTING UP NEW FIGURE ===")
axes, color_id = basic_figure(n, dt)
print("New figure setup completed")

print("=== PLOTTING SECOND STEADY STATE ===")
plot_basic_figure(data, axes, color_id, 10)
print("Second steady state plotted")

print("=== MODIFYING PARAMETERS ===")
print("Setting melt rate (B) to 0")
data.B = 0
print("Setting calving rate (Uc) to 0")
data.Uc = 0
print("Parameters modified")

print("=== STARTING THIRD STEADY STATE SOLVE ===")
print("Solving with B=0 and Uc=0...")
data.steadystate()
print("Third steady state solve completed")

print("=== PLOTTING THIRD STEADY STATE ===")
plot_basic_figure(data, axes, color_id, 0)
print("Third steady state plotted")


#%%
print("=== STARTING GRID REFINEMENT STUDY ===")
print("Setting up figure for grid refinement study...")
axes, color_id = basic_figure(9, 0.01)
print("Grid refinement figure setup completed")

print("=== PLOTTING INITIAL STATE FOR GRID STUDY ===")
plot_basic_figure(data, axes, color_id, 0)
print("Initial state for grid study plotted")

print("=== SETTING UP GRID REFINEMENT SEQUENCE ===")
grid = np.linspace(31,101,8)
print(f"Grid refinement sequence: {grid.astype(int)} points")
print(f"Will test {len(grid)} different grid resolutions")

for j in np.arange(0,len(grid)):
    print(f"\n=== GRID REFINEMENT STEP {j+1}/{len(grid)} ===")
    print(f"Refining grid to {int(grid[j])} points...")
    
    data.regrid(int(grid[j]))
    print(f"Grid refined to {int(grid[j])} points")
    
    print("Setting transient mode...")
    data.transient = 1
    print("Transient mode enabled")
    
    print("Running 10 prognostic steps in transient mode...")
    k = 0
    while k<10:
        print(f"  Transient step {k+1}/10...")
        data.prognostic()
        k += 1
    print("Transient mode steps completed")
    
    print("Disabling transient mode...")
    data.transient = 0
    print("Transient mode disabled")
    
    print("Running final prognostic step...")
    data.prognostic()
    print("Final prognostic step completed")
    
    print("Plotting refined grid state...")
    plot_basic_figure(data, axes, color_id, j+1)
    print(f"Grid refinement step {j+1} completed")

print("=== GRID REFINEMENT STUDY COMPLETED ===")


# #%%
# print('Solving diagnostic equations.')
# data.diagnostic()
# plot_basic_figure(data, axes, color_id, 0)

# #data.param.deps = 0.01 
# #data.diagnostic()
# #plot_basic_figure(data, axes, color_id, 100)



# #%%
# # run prognostic simulations
# start = time.time()
# L_old = data.L
# dL = 1000 # just initiating the change in length with some large value

# t = 0
# k = 0

# print('Solving prognostic equations.')

# #for k in np.arange(1,n):
# while np.abs(dL)>20:      
#     data.dt = 0.25*data.dx*data.L/np.max(data.U)
#     t += data.dt
#     k += 1
#     data.prognostic()
    
#     X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
#     H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))

    
     
#     if (k % 10) == 0:        
#         plot_basic_figure(data, axes, color_id, 0)
#         print('Time: ' + "{:.4f}".format(t) + ' years')   
#         print('Length: ' + "{:.2f}".format(data.L) + ' m')
#         print('Change in length: ' + "{:.2f}".format(data.L-L_old) + ' m') # over 10 time steps
#         print('Volume: ' + "{:.4f}".format(trapz(H, X_)*4000/1e9) + ' km^3')
#         print('H_L: ' + "{:.2f}".format(1.5*data.H[-1]-0.5*data.H[-2]) + ' m') 
#         print('CFL: ' + "{:.4f}".format(data.U[0]*data.dt/data.X[1]))
#         print(' ')
#         dL = data.L-L_old
#         L_old = data.L
#     # data.save(k)

    
# stop = time.time()

# print((stop-start)/60)           

# data.transient = 0
# data.prognostic()
# plot_basic_figure(data, axes, color_id, 100)

# #data.save('steady_B-0pt6_W' + str(Wt) + '_dwdx0.1.pickle')
# #%%
# data.refine_grid(21)

# start = time.time()
# L_old = data.L
# dL = 1000 # just initiating the change in length with some large value

# t = 0
# k = 0

# print('Solving prognostic equations.')

# #for k in np.arange(1,n):
# while np.abs(dL)>20:      
#     data.dt = 0.25*data.dx*data.L/np.max(data.U)
#     t += data.dt
#     k += 1
    
#     data.prognostic()
    
#     X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
#     H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))

    
     
#     if (k % 10) == 0:        
#         plot_basic_figure(data, axes, color_id, 0)
#         print('Time: ' + "{:.4f}".format(t) + ' years')   
#         print('Length: ' + "{:.2f}".format(data.L) + ' m')
#         print('Change in length: ' + "{:.2f}".format(data.L-L_old) + ' m') # over 10 time steps
#         print('Volume: ' + "{:.4f}".format(trapz(H, X_)*4000/1e9) + ' km^3')
#         print('H_L: ' + "{:.2f}".format(1.5*data.H[-1]-0.5*data.H[-2]) + ' m') 
#         print('CFL: ' + "{:.4f}".format(data.U[0]*data.dt/data.X[1]))
#         print(' ')
#         dL = data.L-L_old
#         L_old = data.L
#     # data.save(k)

    
# stop = time.time()

# print((stop-start)/60)           

# data.transient = 0
# data.prognostic()
# plot_basic_figure(data, axes, color_id, 100)

# Python imports
import numpy as np
import matplotlib.pyplot as plt
import rrfc_gdspy as gds
import os
import time, sys, os
#sys.path.append('~/pythonWork/rrfc/lib/python3.8/site-packages/')

# Load constants and design choices. Assumes 2 layer PCB with 1 oz copper and 19 mil total thickness
fc = 5.0e9 # Operating center frequency for electrical length calculations (make this smaller than or equal to the desired operating frequency
z0 = 50 # Desired characteristic impedance of launches
EL = 120 # Desired unit of electrical length in degrees
t = 2.8 # Thickness of conductor in mils
h = 30 # height of conductor above substrate
t_air = 2*(t+h) # thickness of the air above the conductor layer
er = 4.5 # relative permittivity of the substrate material
xEl = 1 # Number of electral length units in the x dimension
yEl = 1 # Numer of electrical length units in the y dimension
#z, e = gds.calcMicrostrip(37.6,h,1000,t,fc,er);
#W, L = gds.synthMicrostrip(fc, z0, 90, 1.4, 19, 4.2)

simulator = 'ADS' # This controls the simulation to be used. Right now there are two valid values 'ADS' or 'EMX'
sim = True # This controls whether a simulation is run or not.
view = False # This controls if the GDS is viewed after each creation 
ports = 4 # For now, code makes either 2, 3 or 4 ports
sides = 2 # For now, code can put ports on 2, 3 or 4 sides, with constraints that are spelled out in rrfc
pixelSize = 10 # the size of the randomized pixel in mils. Typically contrained by a PCB manufacturer.
pathName = '/home/jswalling/pythonWork/rrfc/' # Base path for file creation
for x in range(1,101): # Run 100 iterations of file generation and simulation.
  portPosition, xBoard, yBoard, csv_file, gds_file = gds.randomGDS(ports,sides,xEl, yEl, pixelSize, \
                                                     fc, z0, EL, t, h, er, x, simulator, view)
  data_file = "randomGDS_ports=" + str(ports) + "_sides=" + str(sides) + "_x_y=" + str(xEl) \
              + "_" + str(yEl) + "_seed=" + str(x)
  
  if sim == True:
    print('Port1x = ' + str(portPosition))
    # Import GDS into ADS environment and setup environment for simulation
    libName = 'MyFirstWorkspace'	
    aelName = 'autoloadEMSim.dem'
    gds.createOpenAel(pathName, libName, gds_file, ports, portPosition, aelName)
    
    commands = ['source /software/RFIC/cadtools/cadence/setups/setup-tools',
	        'echo $HPEESOF_DIR',
	        'ads -m ' + pathName + libName + '_wrk/' + aelName + ' &']

    for command in commands:
      os.system(command)
    
    time.sleep(30)
    print('We are still working')

    # Run Momentum Simulation
    os.chdir(pathName + libName + '_wrk/simulation/' + libName + '_lib/RANDOM/layout/emSetup_MoM/')
    commands = ['source /software/RFIC/cadtools/cadence/setups/setup-tools',
	        'adsMomWrapper --dsName=' + data_file + ' --dsDir=' + pathName + 'data/ -O -3D proj proj']

    for command in commands:
      os.system(command)
    
    # Clean up after Momentum Simulation to prepare for next simulation
    aelCloseName = 'autoCloseEMSim.dem'
    gds.createCloseAel(pathName,libName,aelCloseName)

    os.chdir(pathName)
    commands = ['mv ' + csv_file + ' ' + pathName + '/data/pixelMaps/.',
	        'mv ' + gds_file + ' ' + pathName + '/data/gds/.',
	        'source /software/RFIC/cadtools/cadence/setups/setup-tools',
	        'echo $HPEESOF_DIR',
	        'ads -m ' + pathName + libName + '_wrk/' + aelCloseName + ' &',
	        'rm -rf ' + pathName + libName + '_wrk/simulation/*']

    for command in commands:
      os.system(command)

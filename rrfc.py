import csv
import pya
import math
import random
import numpy as np
import gdspy
import sys # read command-line arguments

def ee_HandJ(u: float,
             relPerm: float):
  A = 1.0 + (1.0/49.0)*math.log((math.pow(u,4.0) + math.pow((u/52.0),2.0))/(math.pow(u,4.0) + 0.432)) \
          + (1.0/18.7)*math.log(1.0 + math.pow((u/18.1),3.0));
  B = 0.564*math.pow(((relPerm-0.9)/(relPerm+3.0)),0.053);
  Y= (relPerm+1.0)/2.0 + ((relPerm-1.0)/2.0)*math.pow((1.0 + 10.0/u),(-A*B));
  return Y

def Z0_HandJ(u: float):
  LIGHTSPEED = 2.99792458e8;
  FREESPACEZ0 = 4.0*math.pi*1.0e-7*LIGHTSPEED;
  F = 6.0 + (2.0*math.pi - 6.0)*math.exp(-1*math.pow((30.666/u),0.7528));
  z01 = (FREESPACEZ0/(2*math.pi))*math.log(F/u + math.sqrt(1.0 + math.pow((2/u),2.0)));
  return z01
  
def calcMicrostrip(width: float,
                   height: float,
                   length: float,
                   thick: float,
                   freq: float,
                   relPerm: float):
    """ calculate the impedance of a microstrip line based on dimensions

    Args:
    freq = desired operating frequency in Hz (e.g., 2.4e9)
    imp = desired characteristic impedance (e.g., 50 Ohms)
    length = physical length of conductor (e.g., 1000 mils)
    thick = conductor thickness in mils (e.g., 1.4 mil)
    height = substrate height between conductor and ground plane in mils (e.g., 19 mils)
    height = conductor width in mils (e.g., 38 mils)
    relPerm = relative permittivity of the substrate material
    """                
    u = width/height # ratio of the conductor width to height
    if thick > 0:
      t = thick/height
      u1 = u +(t*math.log(1.0+4.0*math.exp(1)/t/math.pow(1.0/math.tanh(math.sqrt(6.517*u)),2.0)))/math.pi; # from Hammerstad and Jensen
      ur = u +(u1-u)*(1.0+1.0/(math.cosh(math.sqrt(relPerm-1))))/2.0; # from Hammerstad and Jensen
    else:
      u1 = u
      ur = u
    Y = ee_HandJ(ur, relPerm)
    Z0 = Z0_HandJ(ur)/math.sqrt(Y)    
    ereff0 = Y*math.pow(Z0_HandJ(u1)/Z0_HandJ(ur),2.0);
    fn = 1e-9*.00254*freq*height#/1e7 # 1e-9*.00254 converts to GHz*cm
    P1 = 0.27488 + (0.6315 + (0.525 / (math.pow((1 + 0.157*fn),20))) )*u - 0.065683*math.exp(-8.7513*u);
    P2 = 0.33622*(1 - math.exp(-0.03442*relPerm));
    P3 = 0.0363*math.exp(-4.6*u)*(1 - math.exp(-math.pow((fn / 3.87),4.97)));
    P4 = 1 + 2.751*( 1 -  math.exp(-math.pow((relPerm/15.916),8)));
    P = P1*P2*math.pow(((0.1844 + P3*P4)*10*fn),1.5763);
    ereff = (relPerm*P+ereff0)/(1+P); # equavlent relative dielectric constant
    fn = 1e-9*.00254*freq*height#/1e6 # 1e-9*.00254 converts to GHz*cm
    R1 = 0.03891*(math.pow(relPerm,1.4));
    R2 = 0.267*(math.pow(u,7.0));
    R3 = 4.766*math.exp(-3.228*(math.pow(u,0.641)));
    R4 = 0.016 + math.pow((0.0514*relPerm),4.524);
    R5 = math.pow((fn/28.843),12.0);
    R6 = 22.20*(math.pow(u,1.92));
    R7 = 1.206 - 0.3144*math.exp(-R1)*(1 - math.exp(-R2));
    R8 = 1.0 + 1.275*(1.0 -  math.exp(-0.004625*R3*math.pow(relPerm,1.674)*math.pow(fn/18.365,2.745)));
    R9 = (5.086*R4*R5/(0.3838 + 0.386*R4))*(math.exp(-R6)/(1 + 1.2992*R5));
    R9 = R9 * (math.pow((relPerm-1),6))/(1 + 10*math.pow((relPerm-1),6));
    R10 = 0.00044*(math.pow(relPerm,2.136)) + 0.0184;
    R11 = (math.pow((fn/19.47),6))/(1 + 0.0962*(math.pow((fn/19.47),6)));
    R12 = 1 / (1 + 0.00245*u*u);
    R13 = 0.9408*(math.pow( ereff,R8)) - 0.9603;
    R14 = (0.9408 - R9)*(math.pow(ereff0,R8))-0.9603;
    R15 = 0.707*R10*(math.pow((fn/12.3),1.097));
    R16 = 1 + 0.0503*relPerm*relPerm*R11*(1 - math.exp(-(math.pow((u/15),6))));
    R17 = R7*(1 - 1.1241*(R12/R16)*math.exp(-0.026*(math.pow(fn,1.15656))-R15));
    Zc = Z0*(math.pow((R13/R14),R17)) # characteristic impedance
    return Zc, ereff
    
def synthMicrostrip(freq: float,
                    imp: float,
                    eLength: float,
                    thick: float,
                    height: float,
                    relPerm: float):
    """ calculate the dimensions of a microstrip line
    output the width and length of a quarter-wavelength microstrip line in units of mils
    Args:
    freq = desired operating frequency in Hz (e.g., 2.4e9)
    imp = desired characteristic impedance (e.g., 50 Ohms)
    eLength = deisred electrical length (e.g., 90 degrees)
    thick = conductor thickness in mils (e.g., 1.4 mil)
    height = substrate height between conductor and ground plane in mils (e.g., 19 mils)
    relPerm = relative permittivity of the substrate material
    """
    eps_0 = 8.854187817e-12
    mu_0 = 4*math.pi*1e-7
    c = 1/math.sqrt(eps_0*mu_0)
    
    # define some constants
    mur = 1 # relative permeability
    cond = 5.88e7 # conductivity of metal (assumes copper)
    mu = mur * mu_0
    
    lambda_0 = c/freq # wavelength in freespace in m
    lx = 400 # initial length of line in mils
    wmin = 4 # minimum width of conductor in mils
    wmax = 200 # maximum width of conductor in mils
    
    abstol = 1.0e-6
    reltol = 0.1e-6
    maxiters = 50
    
    A = ((relPerm - 1)/(relPerm + 1)) * (0.226 + 0.121/relPerm) + (math.pi/377)*math.sqrt(2*(relPerm+1))*imp
    w_h = 4/(0.5*math.exp(A) - math.exp(-A))
    if w_h > 2:
      B = math.pi*377/(2*imp*math.sqrt(relPerm))
      w_h = (2/math.pi)*(B - 1 - log(2*B - 1) + ((relPerm-1)/(2*relPerm))*(log(B-1) + 0.293 - 0.517/relPerm));
      
    wx = height*w_h
    
    if wx >= wmax:
      wx = 0.95*wmax
      
    if wx <= wmin:
      wx = wmin
      
    wold = 1.01*wx
    zold, erold = calcMicrostrip(wold,height,lx,thick,freq,relPerm);
   
    if zold < imp:
      wmax = wold
    else:
      wmin = wold
    
    iters = 0
    done = 0
    
    while done == 0:
      iters = iters + 1;
      z0, er0 = calcMicrostrip(wx,height,lx,thick,freq,relPerm);
      if z0 < imp:
        wmax = wx
      else:
        wmin = wx
      if abs(z0-imp) < abstol:
        done = 1
      elif abs(wx-wold) < reltol:
        done = 1
      elif iters >= maxiters:
        done = 1
      else:
        dzdw = (z0 - zold)/(wx-wold)
        wold = wx
        zold = z0
        wx = wx - (z0-imp)/dzdw
        if (wx > wmax) or (wx < wmin):
          wx = (wmin + wmax) / 2
    zD, ereff = calcMicrostrip(wx,height,lx,thick,freq,relPerm);
    
    v = c/math.sqrt(ereff)
    l = 39370.0787*eLength/360*v/freq # 39370.0787 converts m to mil
    return wx, l
    
def randomGDS(ports: int,
              sides: int,
              x_el_units: int,
              y_el_units: int,
              pixelSize: int,
              freq: float,
              imp: float,
              eLength: float,
              thick: float,
              height: float,
              relPerm: float, 
              seedNum: int, 
              sim: str,
              view: bool):
  """
  This script will create a rectangular pixel grid that is x_el_units*eLength wide by y_el_units*eLength tall 
  according to the following rules:
  eLength = Electrical Length in Degrees
  x_el_units = Number of electrical length units wide (east-to-west)
  y_el_units = Number of electrical length units tall (north-to-south)
  pixelSize = Size of the pixel in library units (code defaults to mils, but can be modified by changing lib.unit below)
  freq = frequency at which the wavelength is defined
  imp = characteristic impedance of the launch paths from the ports
  thick = thickness of the conductor in library units (code defaults to mils, but can be modified by changing lib.units below)
  height = thickness of the substrate in library units (code defaults to mils, but can be modified by changing lib.unit below) 
           Currently this code assumes that the design is a two layer design with conductors above and below the substrate
  relPerm = relative permittivity of the substrate material
  seedNum = fixed seed number for the random number generator. This allows the same design to be reproduced each time.
  sim = The type of simulator the GDS that is produced is meant to support. There are currently two options, EMX and 
        everything else. The primary reason is the way that ports are defined in EMX, where they must be defined by a
        label in the conductor layer.
  view = Do you want to open up a view window after each gds is created for inspection. This is primarily for de-bugging as new
         features are added. Value must be boolean (either True or False)
  Port definitions must take the following forms, presently no others are acceptable and using any wrong combination will cause
  program to exit with an error message:
  ports = 2, sides = 2: This will create a rectangle with one port (1) on the west side and one port (2) on the west side of 
                        the structure
  ports = 3, sides = 2: This will create a rectangle with one port (1) on the southwest side, one port (2) on the northwest 
                        side and one port (3) on the east side
                        of the structure
  ports = 3, sides = 3: This will create a rectangle with one port (1) on the west side, one port (2) on the east side and one
                        port (3) on the north side of the structure
  ports = 4, sides = 2: This will create a rectangle with one port (1) on the southwest side, one port (2) on the northwest 
                        side, one port (3) on the southeast side, and one port (4) on the northeast side of the structure
  ports = 4, sides = 4: This will create a rectangle with one port (1) on the west side, one port (2) on the north side,
                        one port (3) on the east side, and one port (4) on the south side of the structure
  
  The output of the script is a pixel map, which is a CSV file with ones and zeros in the positions of the pixels in the grid, 
  and a GDS file that can be imported into an EM simulator. It will also output an array with port positions that can be used 
  for placing ports in an ADS simulation. 
  """
  random.seed(seedNum)
  lib = gdspy.GdsLibrary()

  # Set the database unit to 1 mil
  lib.unit = 25.4e-6

  # Create Cell obj
  gdspy.current_library = gdspy.GdsLibrary() # This line of code has to be here to reset the GDS library on every loop
  UNIT = lib.new_cell("RANDOM")

  # Create layer #'s
  if ports == 1:
    l_port1 = {"layer": 1, "datatype": 0}
  if ports == 2:
    l_port1 = {"layer": 1, "datatype": 0}
    l_port2 = {"layer": 2, "datatype": 0}
  if ports == 3:
    l_port1 = {"layer": 1, "datatype": 0}
    l_port2 = {"layer": 2, "datatype": 0}
    l_port3 = {"layer": 3, "datatype": 0}
  if ports == 4:
    l_port1 = {"layer": 1, "datatype": 0}
    l_port2 = {"layer": 2, "datatype": 0}
    l_port3 = {"layer": 3, "datatype": 0}
    l_port4 = {"layer": 4, "datatype": 0}
  l_bottom = {"layer": 10, "datatype": 0}
  l_top = {"layer": 11, "datatype": 0}
  l_sources = {"layer": 5, "datatype": 0}

  # Metal dimensions
  #width_50 = 37.6 # 50 Ohm line width at 2.4 GHz
  #length_90 = 687.5 # Quarter-wavelength line at 2.4 GHz
  width_50, elec_length = synthMicrostrip(freq, imp, eLength, thick, height, relPerm)
#  launch_pixels = 10 # length of the line to connect to structure in number of pixels

  # Pixel Size
  pixel = pixelSize; # Minimum size pixel
  width_launch, length_launch = synthMicrostrip(freq, imp, 30, thick, height, relPerm)
  launch_pixels = math.ceil(length_launch/pixel) # length of the line to connect to structure in number of pixels
  length_launch = launch_pixels*pixel

  # Set horizontal and vertical pixel limits
  y_units = y_el_units # integer units of electrical length
  x_units = x_el_units # integer units of electrical length
  y_dim = math.floor(y_units*elec_length)
  x_dim = math.floor(x_units*elec_length)
  y_pixels = math.ceil(y_dim/pixel)
  x_pixels = math.ceil(x_dim/pixel)
  if ports == 2:
    if sides == 2:
      x_total = (2*launch_pixels + x_pixels)*pixel
      y_total = y_pixels*pixel
      #Define the position of the ports for BEM simulators. This is a one column array with:
      #[port1x, port1y, port2x, port2y, port3x, port3y, port4x, port4y]
      ports = [0, y_total/2, x_total, y_total/2, 0, 0, 0, 0] 
    else: 
      print('For a 2-port network, the number of sides must be equal to 2.')
      quit()
  elif ports == 3:
    if sides == 2:
      x_total = (2*launch_pixels + x_pixels)*pixel
      y_total = y_pixels*pixel
      #Define the position of the ports for BEM simulators. This is a one column array with:
      #[port1x, port1y, port2x, port2y, port3x, port3y, port4x, port4y]
      portPos = [0, y_total/4, 0, 3*y_total/4, x_total, y_total/2, 0, 0]
    elif sides == 3:
      x_total = (2*launch_pixels + x_pixels)*pixel
      y_total = (launch_pixels + y_pixels)*pixel
      #Define the position of the ports for BEM simulators. This is a one column array with:
      #[port1x, port1y, port2x, port2y, port3x, port3y, port4x, port4y]
      portPos = [0, y_total/2, x_total, y_total/2, x_total/2, y_total, 0, 0]
    else:
      print('For a 3-port network, the number of sides must be equal to either 2 or 3.')
      quit()
  elif ports == 4:
    if sides == 2:
      x_total = (2*launch_pixels + x_pixels)*pixel
      y_total = y_pixels*pixel
      #Define the position of the ports for BEM simulators. This is a one column array with:
      #[port1x, port1y, port2x, port2y, port3x, port3y, port4x, port4y]
      portPos = [0, y_total/4, 0, 3*y_total/4, x_total, y_total/4, x_total, 3*y_total/4]
    elif sides == 4:
      x_total = (2*launch_pixels + x_pixels)*pixel
      y_total = (2*launch_pixels + y_pixels)*pixel
      #Define the position of the ports for BEM simulators. This is a one column array with:
      #[port1x, port1y, port2x, port2y, port3x, port3y, port4x, port4y]
      portPos = [0, y_total/2, x_total/2, y_total, x_total, y_total/2, x_total/2, 0]
    else:
      print('For a 4-port network, the number of sides must be equal to either 2 or 4.')
      quit()

  # Draw outline
  outline = gdspy.Rectangle((0, 0), (x_total, y_total), **l_bottom)
  UNIT.add(outline) 
  
  if ports == 2:
    # Add launches: assume a rectangle with port 1 = west, 2 = east
    launch_1 = gdspy.Rectangle((0, y_total/2 - width_50/2), \
               (launch_pixels*pixel, width_50 + y_total/2 - width_50/2), **l_top)
    UNIT.add(launch_1)
    launch_2 = gdspy.Rectangle((x_total, y_total/2 - width_50/2), \
               (x_total - launch_pixels*pixel, width_50 + y_total/2 - width_50/2), **l_top)
    UNIT.add(launch_2)
  elif ports ==3:
    if sides == 2:
      # Add launches: assume rectangle with port 1 = southwest, 2 = northwest, 3 = east
      launch_1 = gdspy.Rectangle((0, y_total/4 - width_50/2), \
                 (launch_pixels*pixel, width_50 + y_total/4 - width_50/2), **l_top)
      UNIT.add(launch_1)
      launch_2 = gdspy.Rectangle((0, 3*y_total/4 - width_50/2), \
                 (launch_pixels*pixel, width_50 + 3*y_total/4 - width_50/2), **l_top)
      UNIT.add(launch_2)
      launch_3 = gdspy.Rectangle((x_total, y_total/2 - width_50/2), \
                 (x_total - launch_pixels*pixel, width_50 + y_total/2 - width_50/2), **l_top)
      UNIT.add(launch_3)
    elif sides == 3:
      # Add launches: assume a rectangle with port 1 = west, 2 = east, 3 = north
      launch_1 = gdspy.Rectangle((0, y_total/2 - width_50/2 - length_launch/2), \
                 (launch_pixels*pixel, width_50 + y_total/2 - width_50/2 - length_launch/2), **l_top)
      UNIT.add(launch_1)
      launch_2 = gdspy.Rectangle((x_total, y_total/2 - width_50/2 - length_launch/2), \
                 (x_total - launch_pixels*pixel, width_50 + y_total/2 - width_50/2 - length_launch/2), **l_top)
      UNIT.add(launch_2)
      launch_3 = gdspy.Rectangle((x_total/2 - width_50/2, y_total), \
                 (width_50 + x_total/2 - width_50/2, y_total - launch_pixels*pixel), **l_top)
      UNIT.add(launch_3)
  elif ports == 4:
    if sides == 2:
      # Add launches: assume a rectangle with port 1 = southwest, 2 = northwest, 3 = southeast, 4 = northeast
      launch_1 = gdspy.Rectangle((0, y_total/4 - width_50/2), \
                 (launch_pixels*pixel, width_50 + y_total/4 - width_50/2), **l_top)
      UNIT.add(launch_1)
      launch_2 = gdspy.Rectangle((0, 3*y_total/4 - width_50/2), \
                 (launch_pixels*pixel, width_50 + 3*y_total/4 - width_50/2), **l_top)
      UNIT.add(launch_2)
      launch_3 = gdspy.Rectangle((x_total, y_total/4 - width_50/2), \
                 (x_total - launch_pixels*pixel, width_50 + y_total/4 - width_50/2), **l_top)
      UNIT.add(launch_3)
      launch_4 = gdspy.Rectangle((x_total, 3*y_total/4 - width_50/2), \
                 (x_total - launch_pixels*pixel, width_50 + 3*y_total/4 - width_50/2), **l_top)
      UNIT.add(launch_4)
    elif sides == 4:
      # Add launches: assume a rectangle with port 1 = west, 2 = north, 3 = east, 4 = south
      launch_1 = gdspy.Rectangle((0, y_total/2 - width_50/2), \
                 (launch_pixels*pixel, width_50 + y_total/2 - width_50/2), **l_top)
      UNIT.add(launch_1)
      launch_2 = gdspy.Rectangle((x_total/2 - width_50/2, y_total), \
                 (width_50 + x_total/2 - width_50/2, y_total - launch_pixels*pixel), **l_top)
      UNIT.add(launch_2)
      launch_3 = gdspy.Rectangle((x_total, y_total/2 - width_50/2), \
                 (x_total - launch_pixels*pixel, width_50 + y_total/2 - width_50/2), **l_top)
      UNIT.add(launch_3)
      launch_4 = gdspy.Rectangle((x_total/2 - width_50/2, 0), \
                 (width_50 + x_total/2 - width_50/2, launch_pixels*pixel), **l_top)
      UNIT.add(launch_4)

  # Add ports and sources to the gds
  if ports == 2:
    if sim == 'ADS':
      port_1 = [(launch_pixels*pixel*0.5, y_total/2 + 1.05*width_50/2), \
                (launch_pixels*pixel*0.5, y_total/2 - 1.05*width_50/2)]
      poly_1 = gdspy.Polygon(port_1, **l_port1)
      UNIT.add(poly_1)
      port_2 = [(x_total - launch_pixels*pixel*0.5, y_total/2 + 1.05*width_50/2), \
                (x_total - launch_pixels*pixel*0.5, y_total/2 - 1.05*width_50/2)]
      poly_2 = gdspy.Polygon(port_2, **l_port2)
      UNIT.add(poly_2)
      source = [(1, y_total/2 + 1.05*width_50/2), (1, y_total/2 - 1.05*width_50/2)]
      poly_3 = gdspy.Polygon(source, **l_sources)
      UNIT.add(poly_3)
    elif sim == 'EMX':
      port_1 = gdspy.Label("p1", (0, y_total/2), "w", layer=11)
      UNIT.add(port_1)
      port_2 = gdspy.Label("p2", (x_total, y_total/2), "e", layer=11)
      UNIT.add(port_2)
    else:
      print('You must choose an available simulator')
      quit()
  elif ports == 3:
    if sides == 2:
      if sim == 'ADS':
        port_1 = [(launch_pixels*pixel*0.5, y_total/4 + 1.05*width_50/2), \
                  (launch_pixels*pixel*0.5, y_total/4 - 1.05*width_50/2)]
        poly_1 = gdspy.Polygon(port_1, **l_port1)
        UNIT.add(poly_1)
        port_2 = [(launch_pixels*pixel*0.5, 3*y_total/4 + 1.05*width_50/2), \
                  (launch_pixels*pixel*0.5, 3*y_total/4 - 1.05*width_50/2)]
        poly_2 = gdspy.Polygon(port_2, **l_port2)
        UNIT.add(poly_2)
        port_3 = [(x_total - launch_pixels*pixel*0.5, y_total/2 + 1.05*width_50/2), \
                  (x_total - launch_pixels*pixel*0.5, y_total/2 - 1.05*width_50/2)]
        poly_3 = gdspy.Polygon(port_3, **l_port3)
        UNIT.add(poly_3)
        source = [(1, y_total/4 + 1.05*width_50/2), (1, y_total/4 - 1.05*width_50/2)]
        poly_4 = gdspy.Polygon(source, **l_sources)
        UNIT.add(poly_4)
      elif sim == 'EMX':
        port_1 = gdspy.Label("p1", (0, y_total/4), "w", layer=11)
        UNIT.add(port_1)
        port_2 = gdspy.Label("p2", (0, 3*y_total/4), "w", layer=11)
        UNIT.add(port_2)
        port_3 = gdspy.Label("p3", (x_total, y_total/2), "e", layer=11)
        UNIT.add(port_3)
      else:
        print('You must choose an available simulator')
        quit()
    elif sides == 3:
      if sim == 'ADS':
        port_1 = [(launch_pixels*pixel*0.5, y_total/2 + 1.05*width_50/2 - length_launch/2), \
                  (launch_pixels*pixel*0.5, y_total/2 - 1.05*width_50/2 - length_launch/2)]
        poly_1 = gdspy.Polygon(port_1, **l_port1)
        UNIT.add(poly_1)
        port_2 = [(x_total - launch_pixels*pixel*0.5, y_total/2 + 1.05*width_50/2 - length_launch/2), \
                  (x_total - launch_pixels*pixel*0.5, y_total/2 - 1.05*width_50/2 - length_launch/2)]
        poly_2 = gdspy.Polygon(port_2, **l_port2)
        UNIT.add(poly_2)
        port_3 = [(x_total/2 + 1.05*width_50/2, y_total - launch_pixels*pixel*0.5), \
                  (x_total/2 - 1.05*width_50/2, y_total -launch_pixels*pixel*0.5)]
        poly_3 = gdspy.Polygon(port_3, **l_port3)
        UNIT.add(poly_3)
        source = [(1, y_total/2 + 1.05*width_50/2 - length_launch/2), \
                  (1, y_total/2 - 1.05*width_50/2 - length_launch/2)]
        poly_4 = gdspy.Polygon(source, **l_sources)
        UNIT.add(poly_4)
      elif sim == 'EMX':
        port_1 = gdspy.Label("p1", (0, y_total/2), "w", layer=11)
        UNIT.add(port_1)
        port_2 = gdspy.Label("p2", (x_total, y_total/2), "e", layer=11)
        UNIT.add(port_2)
        port_3 = gdspy.Label("p3", (x_total/2, y_total), "n", layer=11)
        UNIT.add(port_3)
      else:
        print('You must choose an available simulator')
        quit()
  elif ports == 4:
    if sides == 2:
      if sim == 'ADS':
        port_1 = [(launch_pixels*pixel*0.5, y_total/4 + 1.05*width_50/2), \
                  (launch_pixels*pixel*0.5, y_total/4 - 1.05*width_50/2)]
        poly_1 = gdspy.Polygon(port_1, **l_port1)
        UNIT.add(poly_1)
        port_2 = [(launch_pixels*pixel*0.5, 3*y_total/4 + 1.05*width_50/2), \
                  (launch_pixels*pixel*0.5, 3*y_total/4 - 1.05*width_50/2)]
        poly_2 = gdspy.Polygon(port_2, **l_port2)
        UNIT.add(poly_2)
        port_3 = [(x_total - launch_pixels*pixel*0.5, y_total/4 + 1.05*width_50/2), \
                  (x_total - launch_pixels*pixel*0.5, y_total/4 - 1.05*width_50/2)]
        poly_3 = gdspy.Polygon(port_3, **l_port3)
        UNIT.add(poly_3)
        port_4 = [(x_total - launch_pixels*pixel*0.5, 3*y_total/4 + 1.05*width_50/2), \
                  (x_total - launch_pixels*pixel*0.5, 3*y_total/4 - 1.05*width_50/2)]
        poly_4 = gdspy.Polygon(port_4, **l_port4)
        UNIT.add(poly_4)
        source = [(1, y_total/4 + 1.05*width_50/2), (1, y_total/4 - 1.05*width_50/2)]
        poly_5 = gdspy.Polygon(source, **l_sources)
        UNIT.add(poly_5)
      elif sim == 'EMX':
        port_1 = gdspy.Label("p1", (0, y_total/4), "w", layer=11)
        UNIT.add(port_1)
        port_2 = gdspy.Label("p2", (0, 3*y_total/4), "w", layer=11)
        UNIT.add(port_2)
        port_3 = gdspy.Label("p3", (x_total, y_total/4), "e", layer=11)
        UNIT.add(port_3)
        port_4 = gdspy.Label("p4", (x_total, 3*y_total/4), "e", layer=11)
        UNIT.add(port_4)
      else:
        print('You must choose an available simulator')
        quit()
    elif sides == 4:
      if sim == 'ADS':
        port_1 = [(launch_pixels*pixel*0.5, y_total/2 + 1.05*width_50/2), \
                  (launch_pixels*pixel*0.5, y_total/2 - 1.05*width_50/2)]
        poly_1 = gdspy.Polygon(port_1, **l_port1)
        UNIT.add(poly_1)
        port_2 = [(x_total/2 + 1.05*width_50/2, y_total - launch_pixels*pixel*0.5), \
                  (x_total/2 - 1.05*width_50/2, y_total -launch_pixels*pixel*0.5)]
        poly_2 = gdspy.Polygon(port_2, **l_port2)
        UNIT.add(poly_2)
        port_3 = [(x_total - launch_pixels*pixel*0.5, y_total/2 + 1.05*width_50/2), \
                  (x_total - launch_pixels*pixel*0.5, y_total/2 - 1.05*width_50/2)]
        poly_3 = gdspy.Polygon(port_3, **l_port3)
        UNIT.add(poly_3)
        port_4 = [(x_total/2 + 1.05*width_50/2, launch_pixels*pixel*0.5), \
                  (x_total/2 - 1.05*width_50/2, launch_pixels*pixel*0.5)]
        poly_4 = gdspy.Polygon(port_4, **l_port4)
        UNIT.add(poly_4)
        source = [(1, y_total/2 + 1.05*width_50/2), (1, y_total/2 - 1.05*width_50/2)]
        poly_5 = gdspy.Polygon(source, **l_sources)
        UNIT.add(poly_5)
      elif sim == 'EMX':
        port_1 = gdspy.Label("p1", (0, y_total/2), "w", layer=11)
        UNIT.add(port_1)
        port_2 = gdspy.Label("p2", (x_total/2, y_total), "n", layer=11)
        UNIT.add(port_2)
        port_3 = gdspy.Label("p3", (x_total, y_total/2), "e", layer=11)
        UNIT.add(port_3)
        port_4 = gdspy.Label("p4", (x_total/2, 0), "s", layer=11)
        UNIT.add(port_4)
      else:
        print('You must choose an available simulator')
        quit()

  # Design Random Structure
  pixel_map = np.zeros((x_pixels,y_pixels),dtype=int)
  for x in range(x_pixels):
    for y in range(y_pixels):
      bit = random.random()
      if bit >= 0.5:
      # creates a rectangle for the "on" pixel
        if ports == 2:
          rect = gdspy.Rectangle((0, 0), (pixel, pixel), **l_top).translate(x*pixel + launch_pixels*pixel, y*pixel)
        elif ports == 3:
          if sides == 2:
            rect = gdspy.Rectangle((0, 0), (pixel, pixel), **l_top).translate(x*pixel + launch_pixels*pixel, y*pixel)
          elif sides == 3:
            rect = gdspy.Rectangle((0, 0), (pixel, pixel), **l_top).translate(x*pixel + launch_pixels*pixel, y*pixel)
        elif ports == 4:
          if sides == 2:
            rect = gdspy.Rectangle((0, 0), (pixel, pixel), **l_top).translate(x*pixel + launch_pixels*pixel, y*pixel)
          elif sides == 4:
            rect = gdspy.Rectangle((0, 0), (pixel, pixel), **l_top).translate(x*pixel + launch_pixels*pixel, y*pixel + launch_pixels*pixel)
        UNIT.add(rect)
        pixel_map[x,y] = 1;
      y += 1  
    x += 1

  csvFile = "pixel_map_ports=" + str(ports) + "_sides=" + str(sides) + "_x_y=" + str(x_units) + "_"\
            + str(y_units) + "_seed=" + str(seedNum) + ".csv"
  # Export Pixel Map file
  np.savetxt(csvFile, np.transpose(pixel_map), fmt = '%d', delimiter = ",")

  gdsFile = "randomGDS_ports=" + str(ports) + "_sides=" + str(sides) + "_x_y=" + str(x_units) + "_"\
            + str(y_units) + "_seed=" + str(seedNum) + ".gds"
  # Export GDS
  lib.write_gds(gdsFile)
  if view == True:
    gdspy.LayoutViewer(lib) 

  return portPos, x_total, y_total, csvFile, gdsFile

def createOpenAel(pathName: str,
                  libName: str,
                  gdsName: str,
                  ports: int,
                  portPosition: float,
                  aelName: str):
  """
  This file creates an AEL file that will take a GDS file, input the file into ADS, add ports and prepare the design for simulation.
  The file needs to know the number of ports that will be simulated and the relative position of the ports. These are outputs when 
  the GDS is greated. 
  """
  aelPath = pathName + libName + '_wrk/' + aelName
  with open(aelPath, 'w') as f:
    f.write('/* Begin */\n')
    f.write('de_open_workspace("' + pathName + libName + '_wrk/");\n')
    #f.write('decl macroContext = de_create_new_layout_view("' + libName + '_lib", "cell_1", "layout");\n')
    #f.write('de_show_context_in_new_window(macroContext);\n')
    f.write('de_load_translators_plugin_if_not_loaded();\n')
    f.write('detransdlg_execute_more_options_ok_cb(api_get_current_window(), DE_GDSII_FILE, 1, "' + pathName + gdsName + '");\n')
    f.write('de_import_design(DE_GDSII_FILE, FALSE, "' + pathName + gdsName + '", "' + libName + '_lib", "", NULL);\n')
    f.write('decl macroContext = de_get_design_context_from_name("' + libName + '_lib:RANDOM:layout");\n')
    #f.write('de_close_design("cell_1");\n')
    #f.write('de_delete_cell("' + libName + '_lib", "cell_1");\n') #After import of design cell, delete dummy file that is opened
    f.write('// For an artwork macro: decl macroContext = de_get_current_design_context();\n')
    f.write('de_bring_context_to_top_or_open_new_window(macroContext);\n')
    f.write('db_set_entry_layerid( de_get_current_design_context(), \
             db_find_layerid_by_name( de_get_current_design_context(), "l_top:drawing" ));\n')
    f.write('de_set_grid_snap_type(19);') 
    if ports == 2:
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPostion(1)) + ',' + str(portPostion(2)) \
              + ',' + str(180) + ', db_layerid(39), 1, "P1", 2);\n') 
      # If all goes well, l_top should come in on layer 39 of ADS metal stack, so put the pins on this layer.
      # It may need to adjust if things change. Working to solve.
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPostion(3)) + ',' + str(portPostion(4)) \
              + ',' + str(90) + ', db_layerid(39), 2, "P2", 2);\n')
    elif ports == 3:
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPosition[0]) + ',' + str(portPosition[1]) \
              + ',' + str(180) + ', db_layerid(39), 1, "P1", 2);\n') 
      # If all goes well, l_top should come in on layer 39 of ADS metal stack, so put the pins on this layer.
      # It may need to adjust if things change. Working to solve.
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPosition[2]) + ',' + str(portPosition[3]) \
              + ',' + str(90) + ', db_layerid(39), 2, "P2", 2);\n')
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPosition[4]) + ',' + str(portPosition[5]) \
              + ',' + str(0) + ', db_layerid(39), 3, "P3", 2);\n')
    elif ports == 4:
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPosition[0]) + ',' + str(portPosition[1]) \
              + ',' + str(180) + ', db_layerid(39), 1, "P1", 2);\n') 
      # If all goes well, l_top should come in on layer 39 of ADS metal stack, so put the pins on this layer.
      # It may need to adjust if things change. Working to solve.
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPosition[2]) + ',' + str(portPosition[3]) \
              + ',' + str(90) + ', db_layerid(39), 2, "P2", 2);\n')
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPosition[4]) + ',' + str(portPosition[5]) \
              + ',' + str(0) + ', db_layerid(39), 3, "P3", 2);\n')
      f.write('decl pin = db_create_pin(macroContext, ' + str(portPosition[6]) + ',' + str(portPosition[7]) \
              + ',' + str(-90) + ', db_layerid(39), 4, "P4", 2);\n')   
    f.write('de_open_substrate_window("' + libName + '_lib", "substrate1");\n')
    f.write('decl macroContext = de_get_design_context_from_name("' + libName + '_lib:RANDOM:layout");\n')
    f.write('de_bring_context_to_top_or_open_new_window(macroContext);\n')
    f.write('de_save_oa_design("' + libName + '_lib:RANDOM:layout");\n')
    f.write('decl macroContext = de_get_design_context_from_name("' + libName + '_lib:RANDOM:layout");\n')
    f.write('de_bring_context_to_top_or_open_new_window(macroContext);\n')
    f.write('de_select_all();\n')
    f.write('de_union();\n')
    f.write('decl macroContext = de_get_design_context_from_name("' + libName + '_lib:RANDOM:layout");\n')
    f.write('de_bring_context_to_top_or_open_new_window(macroContext);\n')
    f.write('de_save_oa_design("' + libName + '_lib:RANDOM:layout");\n')
    f.write('de_close_all();\n')
    f.write('dex_em_writeSimulationFiles("' + libName + '_lib", "RANDOM", "emSetup", "simulation/' \
            + libName + '_lib/RANDOM/layout/emSetup_MoM");\n')
    f.write('de_close_workspace_without_prompting();\n')
    f.write('de_exit();\n')

def createCloseAel(pathName: str,
                   libName: str,
                   aelName: str):
  """
  This file creates an AEL file that will clean up the directories after an EM simulation is run and prepare the 
  environment for the next simulation.
  """
  aelPath = pathName + libName + '_wrk/' + aelName
  with open(aelPath, 'w') as f:
    f.write('/* Begin */\n')
    f.write('de_open_workspace("' + pathName + libName + '_wrk/");\n')
    f.write('de_delete_cellview("' + libName + '_lib", "RANDOM", "layout");\n')
    f.write('de_close_workspace_without_prompting();\n')
    f.write('de_exit();\n')

# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage

def t2p(a):
    """  Converts a masked TUFLOW velocity direction (va_) raster in degrees to polar coorindates in radians.
         TUFLOW format - North = 0°,  East = 90°, South = 180°/-180°, West = -90°
         Polar format  - North = 90°, East = 0°,  South = 270°,       West = 180°
         Returns a masked array of radians.
    """

    part_a = (-1 * (a - 90)) * (np.absolute(a) <= 90) # the northern half
    part_b = (-1 * a + 90) * (np.absolute((a < 0) * a) > 90) # the south-western quarter
    part_c = (-1 * a + 360 + 90) * (np.absolute((a > 0) * a) > 90) # the south-eastern quarter
    
    return np.radians(part_a + part_b + part_c)
                          
def inflows(x):
    n = min(0,(np.sin(x) * np.array([1,0,0,0])).sum())
    s = max(0,(np.sin(x) * np.array([0,0,0,1])).sum())
    e = min(0,(np.cos(x) * np.array([0,0,1,0])).sum())
    w = max(0,(np.cos(x) * np.array([0,1,0,0])).sum())                                    
    return n + s + e + w

#tuflow = np.array([ 90, 0,  -90,  180, -180, 45, -45, -135, 135]) # tuflow output
#polar = np.array([  0,  90,  180, 270,  270, 45, 135,  225, 315]) # correct polar degrees

null_value = -999

va = np.array(([-999,-999,-999,-999,-999],
              [-999,-999,   0,-999,-999],
              [-999,-999, 180,-999,-999],
              [-999,-999, 180,-999,-999],
              [-999,-999,-999,-999,-999])) 

va_ = np.ma.masked_array(va, (va == null_value))   

footprint = np.array([[0,1,0],
                      [1,0,1],
                      [0,1,0]])    
            
res = ndimage.generic_filter(t2p(va_), inflows, footprint=footprint)
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage

np.set_printoptions(precision=3)

def t2p(a):
    """  Converts a masked TUFLOW velocity direction (va_) raster in degrees to polar coorindates in radians.
         TUFLOW format - North = 0°,  East = 90°, South = 180°/-180°, West = -90°
         Polar format  - North = 90°, East = 0°,  South = 270°,       West = 180°
         Returns a masked array of radians. """

    part_a = (-1 * (a - 90)) * (np.absolute(a) <= 90) # the northern half
    part_b = (-1 * a + 90) * (np.absolute((a < 0) * a) > 90) # the south-western quarter
    part_c = (-1 * a + 360 + 90) * (np.absolute((a > 0) * a) > 90) # the south-eastern quarter
    
    return np.radians(part_a + part_b + part_c)                   
  
def t2p_n(a):
    """  Converts a non-masked TUFLOW velocity direction (va) raster in degrees to polar coorindates in radians.
         TUFLOW format - North = 0°,  East = 90°, South = 180°/-180°, West = -90°
         Polar format  - North = 90°, East = 0°,  South = 270°,       West = 180°
         Returns a non-masked array of radians. """

    null = a == -999
    not_null = np.logical_not(null)

    part_a = (-1 * (a - 90)) * (np.absolute(a) <= 90) # the northern half
    part_b = (-1 * a + 90) * (np.absolute((a < 0) * a) > 90) # the south-western quarter
    part_c = (-1 * a + 360 + 90) * (np.absolute((a > 0) * a) > 90) # the south-eastern quarter
    
    return not_null * np.radians(part_a + part_b + part_c) + null * a                                                                      
  
def inflows(x):
    """ Filter function for masked array with no -999 values.
        Calculates flows towards cell from neighbours. """
    n = min(0,(np.sin(x) * np.array([1,0,0,0])).sum()) * -1
    s = max(0,(np.sin(x) * np.array([0,0,0,1])).sum())
    e = min(0,(np.cos(x) * np.array([0,0,1,0])).sum()) * -1
    w = max(0,(np.cos(x) * np.array([0,1,0,0])).sum())                                    
    return n + s + e + w
                                                                    
def inflows_n(x):
    """ Filter function for array with -999 values.
        Calculates flows towards cell from neighbours. """
    (n, s, e, w) = 0, 0, 0, 0
    u = (x * np.array([1,0,0,0])).sum()
    d = (x * np.array([0,0,0,1])).sum()
    l = (x * np.array([0,0,1,0])).sum()
    r = (x * np.array([0,1,0,0])).sum()
    if u != -999:
        n = min(0,(np.sin(u))) * -1   
    if d != -999:
        s = max(0,(np.sin(d)))
    if l != -999:
        e = min(0,(np.cos(l))) * -1
    if r != -999:
        w = max(0,(np.cos(r)))                                    
    return n + s + e + w    

#tuflow = np.array([ 90, 0,  -90,  180, -180, 45, -45, -135, 135]) # tuflow output
#polar = np.array([  0,  90,  180, 270,  270, 45, 135,  225, 315]) # correct polar degrees

null_value = -999

va = np.array(([-999,-999,-999,-999,-999],
               [-999,-999,   0,-999,-999],
               [-999,  90,   0, -90,-999],
               [-999,-999,   0,-999,-999],
               [-999,-999,-999,-999,-999])) 

va_ = np.ma.masked_array(va, (va == null_value))   

flow_ac = np.ones_like(va)

footprint = np.array([[0,1,0],
                      [1,0,1],
                      [0,1,0]])    

rads_1 = t2p_n(va)            
                                    
res_1 = np.around(ndimage.generic_filter(rads_1, inflows_n, footprint=footprint), 10)
flow_ac_1 = flow_ac + res_1

rads_2 = np.where(res_1 != 0, rads_1, -999)

res_2 = np.around(ndimage.generic_filter(rads_2, inflows_n, footprint=footprint), 10)
flow_ac_2 = flow_ac_1 + res_2

rads_3 = np.where(res_2 != 0, rads_1, -999)

res_3 = np.around(ndimage.generic_filter(rads_3, inflows_n, footprint=footprint), 10)
flow_ac_3 = flow_ac_2 + res_3
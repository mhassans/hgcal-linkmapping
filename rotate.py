#!/usr/bin/env python3
import pandas as pd
import numpy as np

class uvCoordinate:
    def __init__(self, u, v):
        self.u = u
        self.v = v
    def convert_to_cube(self):
        x = self.u
        z = -self.v
        y= -x-z
        return cubeCoordinate(x, y, z)
    def coords(self):
        return [self.u,self.v]

        
class cubeCoordinate:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __sub__(self,cube):
        return cubeCoordinate(self.x-cube.x,self.y-cube.y,self.z-cube.z)
    def __add__(self,cube):
        return cubeCoordinate(self.x+cube.x,self.y+cube.y,self.z+cube.z)
    def rotate(self,angle):
        n_times = int(angle/60)
        current = [self.x,self.y,self.z]
        new = current[:]
        for n in range(n_times):
            new[0] = -current[1]
            new[1] = -current[2]
            new[2] = -current[0]
            current = new[:]
        return cubeCoordinate(new[0],new[1],new[2])
    def rotatePoint(self,angle):
        n_times = int(angle/60)
        current = [self.x,self.y,self.z]
        new = current[:]
        for n in range(n_times):
            new[0] = -current[1]
            new[1] = -current[2]
            new[2] = -current[0]
            current = new[:]
        return cubeCoordinate(new[0],new[1],new[2])
    def convert_to_uv(self):
        return uvCoordinate(self.x, -self.z)    
    def coords(self):
        return [self.x,self.y,self.z]

def rotate_cell_around(cell,centre):
    P = uvCoordinate(cell[0],cell[1])
    C = uvCoordinate(centre[0],centre[1])
    
    cubeP = P.convert_to_cube()
    cubeC = C.convert_to_cube()

    P_from_C = cubeP-cubeC
    R_from_C = P_from_C.rotate(120)
    cubeR = R_from_C+cubeC
    final = cubeR.convert_to_uv()
    uv_rotated = [np.round(final.u), np.round(final.v)]
    return uv_rotated

    

def rotate(u,v,layer,sector):

    uv = [u,v]
    centre = [0,0]
    

    if (layer > 28):
        if (layer % 2) == 0:
            centre = [-1/3,(-2/3)]
        else:
            centre = [+1/3,(+2/3)]

    uv_rotated = []
    if ( sector == 0 ):
        return uv
    elif ( sector == 1 ):
        return rotate_cell_around(uv,centre)
    elif ( sector == 2 ):
        return rotate_cell_around(rotate_cell_around(uv,centre),centre)

    #print("original = ",uv)    
    #print("rotated = ", uv_rotated)

    return uv_rotated

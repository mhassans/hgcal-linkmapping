#!/usr/bin/env python3
import sys

def rotate_cell_around_centre_general(cell,centre):

    rotated = [centre[0]+centre[1]-cell[1],-centre[0]+2*centre[1]+cell[0]-cell[1]]
    return rotated

def rotate_cell_CEE(cell):

    rotated = [-cell[1],cell[0]-cell[1]]
    return rotated

def rotate_cell_CEH_Odd(cell):

    rotated = [-1-cell[1],-1+cell[0]-cell[1]]
    return rotated

def rotate_cell_CEH_Even(cell):

    rotated = [1-cell[1],1+cell[0]-cell[1]]
    return rotated

def rotate(u,v,layer,sector):
    uv = [u,v]

    if ( sector == 0 ):
        return uv
    
    if (layer > 28):
        if (layer % 2) == 0:

            if ( sector == 1 ):
                return rotate_cell_CEH_Even(uv)
            elif ( sector == 2 ):
                return rotate_cell_CEH_Even(rotate_cell_CEH_Even(uv))

        else:

            if ( sector == 1 ):
                return rotate_cell_CEH_Odd(uv)
            elif ( sector == 2 ):
                return rotate_cell_CEH_Odd(rotate_cell_CEH_Odd(uv))

    else:
        if ( sector == 1 ):
            return rotate_cell_CEE(uv)
        elif ( sector == 2 ):
            return rotate_cell_CEE(rotate_cell_CEE(uv))

            
def rotate_to_sector_0(u,v,layer):

    sector = 0
    uv = [u,v]

    if (layer > 28):

        if (layer % 2) == 0: #Even CEH Layers

            if ( u > 0 and v > 0 ): #sector 0
                return uv,sector
            elif ( u <= 0 and v > u ): #sector 1
                sector = 1
                return rotate_cell_CEH_Even(rotate_cell_CEH_Even(uv)),sector
            else: #sector 2
                sector = 2
                return rotate_cell_CEH_Even(uv),sector
            
        else: #Odd CEH Layers
                
            if ( u >= 0 and v >= 0 ): #sector 0
                return uv,sector
            elif ( u < 0 and v >= u ): #sector 1
                sector = 1
                return rotate_cell_CEH_Odd(rotate_cell_CEH_Odd(uv)),sector
            else: #sector 2
                sector = 2
                return rotate_cell_CEH_Odd(uv),sector

    else: #CEE

        if ( u > 0 and v >= 0 ): #sector 0
            return uv,sector
        elif ( u <= 0 and v > u ): #sector 1
            sector = 1
            return rotate_cell_CEE(rotate_cell_CEE(uv)),sector
        else: #sector 2
            sector = 2
            return rotate_cell_CEE(uv),sector

#Main function for standalone use

# def main():
    
#     u = int(sys.argv[1])
#     v = int(sys.argv[2])
#     layer = int(sys.argv[3])

#     uv = rotate_to_sector_0(u,v,layer)
#     print ( "python: " +  str(uv[0]) + "," + str(uv[1]) )


# main()

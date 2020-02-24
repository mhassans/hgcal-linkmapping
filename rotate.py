#!/usr/bin/env python3

def rotate_cell_around_centre_general(cell,centre):

    rotated = [centre[0]+centre[1]-cell[1],-centre[0]+2*centre[1]+cell[0]-cell[1]]
    return rotated

def rotate_cell_CEE(cell):

    rotated = [-cell[1],cell[0]-cell[1]]
    return rotated

def rotate_cell_CEH_Even(cell):

    rotated = [-1-cell[1],-1+cell[0]-cell[1]]
    return rotated

def rotate_cell_CEH_Odd(cell):

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

            


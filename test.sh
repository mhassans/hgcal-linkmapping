#!/bin/bash

#Test you get the same rotation in c++ and python implementation

for LAYER in {1,10,20,27,28,29,30,31,32}
do

    echo "LAYER = " $LAYER

    for U in {-10..10}
    do
	for V in {-10..10}
	do
	    echo "ORIGINAL = " $U","$V
	    ./rotate $U $V $LAYER
	    ./rotate.py $U $V $LAYER

	done
    done
done


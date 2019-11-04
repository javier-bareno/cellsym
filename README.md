# cellsym
Phenomenological lithium-ion cell simulator.

In this repository I try to develop a set of Python tools to simulate tests of lithium-ion batteries. The idea is to use experimental V-Q characteristics of electrodes to simulate the idela behavior of batteries. Then, introduce non-ideal behavior phenomenologically; e.g., by reducing the amount of active material at certain cycle positions.

See /SCA_LION cells/SCA-LION cells.pptx for a description of the model and samples of output data.

The main code is in bin/cellsym.py 

There is an example of use in bin/Cell simulations.ipynb

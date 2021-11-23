# A full-parallel implementation of Self-Organizing Map on hardware

This repository contains the Matlab algorithm used to generate the data sets considered in the validation and evaluation of the above paper. The software version of the SOM algorithm (also implemented on Matlab) used to compare with the hardware implementation, is also available.

Replicating the implementation (simulink simulation - developed on Matlab 2012 with System Generator 14):
1. Run the _ConfigValidate_ script. This will generate the data sets, the software version of the SOM, and all the hardware parameters.
2. Run the hardware version desired, named Parallel_n**x**. The 'n' refers to the number of neurons.
3. Run the _FinalHardwarePlot_ script to plot the results. 

To run it on the FPGA it is necessary to generate the bitstream, which can be done using the Xilinx Token. 

# Article
The article is available here: https://doi.org/10.1016/j.neunet.2021.05.021

# Reference

Leonardo A. Dias, Augusto M.P. Damasceno, Elena Gaura, Marcelo A.C. Fernandes,
A full-parallel implementation of Self-Organizing Maps on hardware,
Neural Networks,
Volume 143,
2021,
Pages 818-827,
ISSN 0893-6080,
https://doi.org/10.1016/j.neunet.2021.05.021.
(https://www.sciencedirect.com/science/article/pii/S0893608021002173)

# Funding

This study was funded in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) Finance Code 001.

The authors would like to acknowledge the financial support of the Engineering and Physical Science Research Council (EPSRC) for funding the EnergyREV project (EP/S031863/1).


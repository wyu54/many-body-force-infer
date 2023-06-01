# many-body-force-infer
========
Infer interaction, confinement and damping force from many-body trajectories. 
The trajectories should be given in a csv file, with columns frame, particle index, and particles' positions. An example is given in 0.75Pa15p_cleared.csv.  
The code first pre-process the data (the get_data) function which transforms the data into a 3d tensor (the middle dimension puly shifts the data by delta_t and stack together)
plus a 2d tensor (w convolve v) for the input of data and a 2d tensor (w convolve a) for the output.
For implementing the model, please check demonstration.ipynb

References
==========

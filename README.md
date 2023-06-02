# many-body-force-infer
Our project provides a Keras model for predicting particle dynamics in a system based on given input conditions. The model utilizes Tensorflow's capabilities and Keras's simplicity, offering a sophisticated solution for studying complex systems.

# requirements
tensorflow 2.4
sympy
pandas
jupyter notebook 

# tutorial
Your data for particle trajectories should be formatted in a .csv file. This file must include columns for the frame index, particle index, and the particle's positional coordinates (x, y, z). One example of data file is shown in 0.75Pa15p_cleared.csv, which is experimental particles' trajectories in dusty plasma, tracked by our own 3D imaging system. 

To begin using the model, start by preprocessing your data with the get_data function. This step is important as it involves the calculation of various particle descriptors.

The processed data will include two components: 'X' and 'Y'. 'X' is a tuple containing two elements: a 3D tensor representing particle positions, and a 2D tensor representing velocity. 'Y', on the other hand, is a 2D tensor representing particle acceleration.

Once you have your preprocessed data, you can proceed to train the model. Details on how to train the model and predict parameters such as gamma, interaction force, and confinement force are provided in the demonstration.ipynb notebook. This resource provides step-by-step instructions to guide you through the entire process.

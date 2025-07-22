[![DOI](https://zenodo.org/badge/648375177.svg)](https://doi.org/10.5281/zenodo.15866619)

# many-body-force-infer
This code repository serves as a supplement to the paper (https://arxiv.org/abs/2310.05273) "Physics-tailored Machine Learning Reveals Unexpected Physics in Dusty Plasmas", offering a comprehensive implementation of our model for inferring and predicting the forces on particles in a dusty plasma, or other many body system. The forces on particles are restricted to be pairwise and two-dimensional, although they can depend on a third spatial dimension.
# requirements
numpy 1.21.5

matplotlib 3.3.2

tensorflow 2.4

sympy 1.10.1

pandas 1.2.1

jupyter notebook 

# tutorial
To effectively utilize the model, ensure that your particle trajectory data is properly formatted in a .csv file. This file should consist of columns representing the frame index, particle index, and the particle's positional coordinates (x, y, z) in sequential order. For instance, you can refer to the example data file named 0.75Pa15p_cleared.csv, which contains experimental particle trajectories obtained from our dedicated 3D imaging system (for more details, please refer to our paper).
​
To initiate the modeling process, you will need to preprocess your data using the get_data function. This preprocessing step is crucial as it involves the calculation of various particle descriptors, enabling the subsequent analysis.
​
The preprocessed data comprises two main components: 'X' and 'Y'. 'X' is a tuple that includes a 3D tensor representing particle positions and a 2D tensor representing particle velocity. 'Y', on the other hand, is a 2D tensor representing particle acceleration, which is inferred directly from the provided data.
​
Once you have obtained the preprocessed data, you can proceed to train the model. The demonstration.ipynb notebook provides comprehensive instructions and a practical example on how to train the model and predict essential parameters such as gamma, interaction force, and confinement force. This resource serves as a step-by-step guide, ensuring a seamless experience throughout the entire process.

Some example data are uploaded as csv file. the columns are frame (frame index), particle (particle index), x, y, z, and mass (brightness of the particles in the camera)

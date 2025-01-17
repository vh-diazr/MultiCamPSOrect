# MultiCamPSOrect
### Python Implementation for Multi-Camera Stereo Image Rectification presented in

> **Diaz-Ramirez, V.H.; Gonzalez-Ruiz, M.; Juarez-Salazar, R.; Cazorla, M.**  
> _Reliable Disparity Estimation Using Multiocular Vision with Adjustable Baseline._  
> _Sensors 2025, 25, 21._  
> [https://doi.org/10.3390/s25010021](https://doi.org/10.3390/s25010021)  


### General Description:
<p>Stereo vision is an accessible and effective technology for three-dimensional (3D) imaging. The 3D structure of a scene is obtained through triangulation from the disparity map of captured stereo images. This disparity map is composed by the horizontal differences of corresponding points in the stereo images.

To ensure accurate disparity estimation and minimize distortion, image rectification must be performed beforehand to satisfy the epipolar constraint.

For binocular images, rectification typically involves two steps:

1. Estimating a rectifying homography for each camera.
2. Applying a projective transformation to each image using the corresponding homography.

However, rectifying multiocular systems with more than two cameras introduces greater complexity.</p> 

<p>This repository provides a Python implementation for rectifying multi-camera stereo images using Particle Swarm Optimization (PSO). The implementation supports multi-camera arrays with two, three, or four cameras.</p>

### Citation:

	@article{rectMultiCamPSwo:2025,		
	  title = "Reliable Disparity Estimation Using Multiocular Vision with Adjustable Baseline",
	  author = "Diaz-Ramirez, Victor H. and Gonzalez-Ruiz, Martin and Juarez-Salazar, Rigoberto and Cazorla, Miguel",
	  journal = "Sensors",
	  volume = "25",
    number = "1"   
	  year = "2025",
	} 

### Requirements:

| Language and libraries | Tested version |
|------------------------| --- |
| Python                 | 3.12.2 |
| opencv-python          | 4.9.0 |
| numpy | 1.26.4 |
| numba | 0.59.1 |
| pyswarms | 1.3.0 |
| matplotlib | 3.8.3 |
 
### Usage:
  - python3 mainMultiCamPSOrect.py

### Configuration parameters:

| Parameter | Description                                                                                           |
| --------- |-------------------------------------------------------------------------------------------------------|
| Ncam = 3 | Specify the number of cameras in the multiocular vision system. Feasible values for Ncam: 2, 3, or 4. |
| dir = 'testimgs/' | Directory path for the input multiocular image folder.                                                |
| imname = 'exp02' | File names of the input multiocular images.                                                           |
| Np0 = 250 | Number of desired corresponding points to detect.                                                     |
| deg = 60 | Set optimization bounds for rotation angles within the interval [-deg, deg].                          |
| scl = 0.3 | Set optimization bounds for scaling within the interval [1 - scl, 1 + scl].                           |
| tx = 0.3 | Set optimization bounds for x-translation within the interval [-tx, tx].                              |
| ty = 0.3 | Set optimization bounds for y-translation within the interval [-ty, ty].                              | 
| w = 0.95 | Set optimization trade-off coefficient for epipolar constraint and distortion minimization. |

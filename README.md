# MultiCamPSOrect
### Python Implementation for Multi-Camera Stereo Image Rectification

<p>
Stereo vision is an effective and easy access technology for three-dimensional (3D) imaging.
The 3D distribution of a scene is obtained by triangulation from the disparity map of the captured stereo images. 
The disparity map, is composed by the horizontal location differences of all corresponding points in the stereo images. 
However, it is essential to be performed previously a rectification of the captured images to satisfy the epipolar constraint while minimizing distortion. 
Conventionally, image rectification is carried out in binocular images by firstly estimating a rectifying homography for each camera, and secondly, by performing a projective transformation on each captured image using a corresponding homography. 
However, when a multi-ocular imaging system with more than two cameras is used the rectification becomes considerable more challenging.</p> 

<p>This repository offers a Python implementation for rectifying multi-camera stereo images using Particle Swarm Optimization (PSO). 
This implementation allows users to rectify images from multi-camera arrays, comprising two, three, and four cameras.
</p>

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

| Parameter | Description |
| --------- | ----------- |
| Ncam = 3 | Number of cameras of the multiocular vision system. Feasible values Ncam = 2,3,4 |
| dir = 'testimgs/' | Folder containing the input multiocular image. |
| imname = 'exp02' | Name of the input multiocular images. |
| Np0 = 250 | Number of desired corresponding points to be detected. |
| deg = 60 | Specify the optimization bounds for rotation angles in the interval [-deg,deg]. |
| scl = 0.3 | Specify the optimization bounds for scaling in the interval [1-scl,1+scl]. |
| tx = 0.3 | Specify the optimization bounds for x-translation in the interval [-tx,tx]. |
| ty = 0.3 | Specify the optimization bounds for y-translation in the interval [-ty,ty]. | 
| w = 0.95 | Optimization coefficient for trade-off between epipolar constraint and distortion minimization. |
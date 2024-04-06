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
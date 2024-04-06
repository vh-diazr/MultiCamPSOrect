import cv2
import numpy as np
import matplotlib.pylab as plt
from pyswarms.single.global_best import GlobalBestPSO
import projective_lib as proj
import time
############################################################################
# Define functions or code blocks for each case
def twoCam(**kwargs):
    print("Two-camera image rectification")
    from objective_function_2Cam import fitnessFUN2Cam
    ######################################################
    '''Parameters for the objective function and PSwO'''
    ftol = 1e-7         # tolerance of the fitness value for termination
    ftol_iter = 40      # number of iterations that fitness-value not improve more than 'ftol'
    max_iter = 1500     # maximum number of allowed iteration for PSwO algorithm
    psw_opts = {'c1': 0.5, 'c2': 0.3, 'w': 0.9} # coefficients of the PSwO algorithm
    fitmin = 2e-2       # minimum allowed fitness-value after PSwO termination
    maxtrials = 5       # maximum number of trials to obtain a solution fitness<=fitmin
    ######################################################
    Nvar = 6 * Ncam     # number of parameters to be optimized (six per camera)
    Npart = 10 * Nvar   # number of particles for the PSwO algorithm

    '''Reading stereo images to be rectified'''
    print('Reading stereo images to be rectified ', end='')
    start_time = time.time()
    img1, imgg1 = proj.load_images(dir + imname + '_1.png')
    img2, imgg2 = proj.load_images(dir + imname + '_2.png')
    MNc = img1.shape  # Shape of input images (rows,columns,channels)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(done in ", round(elapsed_time, 2), "seconds)")

    '''Detection of corresponding points in the input images (SIFT method)'''
    print('Detecting corresponding points using SIFT: ', end='')
    start_time = time.time()
    matchedPoints_c21, matchedPoints_c12, Np = proj.interest_points(imgg1, imgg2, Np0)
    pts_c12 = np.array([kp.pt for kp in matchedPoints_c12])  # List of coordinates of detected points in image 1
    pts_c21 = np.array([kp.pt for kp in matchedPoints_c21])  # List of coordinates of detected points in image 2
    # show_matched_features(img1, img2, matchedPoints_c12, matchedPoints_c21)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(detected ", Np, " points in ", round(elapsed_time, 2), "seconds)")

    '''bounds of the search space for rectification parameters'''
    gamma_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    theta_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    phi_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    f_bnd = [1 - scl, 1 + scl]  # Scaling parameter: (<1 reduce) (>1 increase) (=1 unscaling)
    tx_bnd = [-tx, tx]  # translation in x-direction (in normalized coordinates)
    ty_bnd = [-ty, ty]  # translation in y-direction (in normalized coordinates)
    #######################################################################################

    # Specification of lower and upper bounds for particle swarm optimization
    lb = np.array([gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0],
                   gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0]])
    ub = np.array([gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1],
                   gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1]])

    '''Initialization of the Particle Swarm Optimization optimizer'''
    Kn = proj.gcoord(np.array(MNc))  # Transformation matrix Kn (3x3) for coordinate normalization
    bounds = (lb, ub)
    '''Additional parameters passed to the fitness function'''
    kwargs = {"pts_c12": pts_c12, "pts_c21": pts_c21, 'Kn': Kn, "MNc": MNc, "w": w}
    '''Computation of projective transformation parameters for stereo rectification'''
    fitness = 1 # worse-case fitness value
    trials = 0  # reset of PSwO optimization trials
    print('Computing rectifying homographies: ', end='')
    start_time = time.time()
    while (fitness > fitmin) and (trials <= maxtrials):
        optimizer = GlobalBestPSO(n_particles=Npart, dimensions=Nvar, options=psw_opts, bounds=bounds, ftol=ftol,
                                  ftol_iter=ftol_iter)
        fitness, pose = optimizer.optimize(fitnessFUN2Cam, iters=max_iter, verbose=True, **kwargs)
        trials += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(done after',trials, 'trials in', round(elapsed_time, 2), 'seconds: fitness=', round(fitness, 3), ')')

    print('Computing rectified images by projective transformation: ', end='')
    start_time = time.time()
    ang = [pose[0], pose[1], pose[2]]
    pix = [pose[3], pose[3], 0]
    txy = [pose[4], pose[5]]
    Sr1 = np.concatenate((ang, pix, txy))
    Gr1 = proj.homography8dgf(Sr1)

    ang = [pose[6], pose[7], pose[8]]
    pix = [pose[9], pose[9], 0]
    txy = [pose[10], pose[11]]
    Sr2 = np.concatenate((ang, pix, txy))
    Gr2 = proj.homography8dgf(Sr2)

    e_a = np.empty((0, 0))
    img1r = proj.proj_transform_numba(img1, Gr1, e_a, 'k')
    img2r = proj.proj_transform_numba(img2, Gr2, e_a, 'k')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(done in", round(elapsed_time, 2), "seconds)")

    fig, axs = plt.subplots(2, 2)
    axs[0,0].imshow(img1)
    axs[0,1].imshow(img2)
    axs[1,0].imshow(img1r)
    axs[1,1].imshow(img2r)
    fig.text(0.5, 0.98, 'Input unrectified images', ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.50, 'Output rectified images', ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.show()
def threeCam(**kwargs):
    print("Image rectification of three cameras")
    # Code for case 2
    from objective_function_3Cam import fitnessFUN3Cam
    ######################################################
    '''Parameters for the objective function and PSwO'''
    ftol = 1e-7         # tolerance of the fitness value for termination
    ftol_iter = 40      # number of iterations that fitness-value not improve more than 'ftol'
    max_iter = 2000     # maximum number of allowed iteration for PSwO algorithm
    psw_opts = {'c1': 0.5, 'c2': 0.3, 'w': 0.9} # coefficients of the PSwO algorithm
    fitmin = 1e-2       # minimum allowed fitness-value after PSwO termination
    maxtrials = 5       # maximum number of trials to obtain a solution fitness<=fitmin
    ######################################################
    Nvar = 6 * Ncam     # number of parameters to be optimized (six per camera)
    Npart = 12 * Nvar   # number of particles for the PSwO algorithm

    '''Reading stereo images to be rectified'''
    print('Reading stereo images to be rectified ', end='')
    start_time = time.time()
    img1, imgg1 = proj.load_images(dir + imname + '_1.png')
    img2, imgg2 = proj.load_images(dir + imname + '_2.png')
    img3, imgg3 = proj.load_images(dir + imname + '_3.png')
    MNc = img1.shape  # Shape of input images (rows,columns,channels)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(done in ", round(elapsed_time, 2), "seconds)")

    '''Detection of corresponding points in the input images (SIFT method)'''
    print('Detecting corresponding points using SIFT: ', end='')
    start_time = time.time()
    matchedPoints_c21, matchedPoints_c12, Np12 = proj.interest_points(imgg1, imgg2, Np0)
    matchedPoints_c31, matchedPoints_c13, Np13 = proj.interest_points(imgg1, imgg3, Np0)
    pts_c12 = np.array([kp.pt for kp in matchedPoints_c12])  # List of coordinates of detected points in image 1
    pts_c21 = np.array([kp.pt for kp in matchedPoints_c21])  # List of coordinates of detected points in image 2
    pts_c13 = np.array([kp.pt for kp in matchedPoints_c13])  # List of coordinates of detected points in image 1
    pts_c31 = np.array([kp.pt for kp in matchedPoints_c31])  # List of coordinates of detected points in image 2
    Np = min([Np12, Np13])      # update the number of detected matched points
    pts_c12 = pts_c12[:Np, :]
    pts_c21 = pts_c21[:Np, :]
    pts_c13 = pts_c13[:Np, :]
    pts_c31 = pts_c31[:Np, :]
    # show_matched_features(img1, img2, matchedPoints_c12, matchedPoints_c21)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(detected ", Np, " points in ", round(elapsed_time, 2), "seconds)")

    '''Specification of bounds of the search space for rectification parameters'''
    gamma_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    theta_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    phi_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    f_bnd = [1 - scl, 1 + scl]  # Scaling parameter: (<1 reduce) (>1 increase) (=1 unscaling)
    tx_bnd = [-tx, tx]          # translation in x-direction (in normalized coordinates)
    ty_bnd = [-ty, ty]          # translation in y-direction (in normalized coordinates)

    # Specification of lower and upper bounds for particle swarm optimization
    lb = np.array([gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0],
                   gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0],
                   gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0]])
    ub = np.array([gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1],
                   gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1],
                   gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1]])

    '''Initialization of the Particle Swarm Optimization optimizer'''
    Kn = proj.gcoord(np.array(MNc))  # Transformation matrix Kn (3x3) for coordinate normalization
    bounds = (lb, ub)
    '''Additional parameters passed to the fitness function'''
    kwargs = {"pts_c12": pts_c12, "pts_c21": pts_c21, "pts_c13": pts_c13, "pts_c31": pts_c31, "Kn": Kn, "MNc": MNc,
              "w": w}
    '''Computation of projective transformation parameters for stereo rectification'''
    fitness = 1
    trials = 0
    print('Computing rectifying homographies: ', end='')
    start_time = time.time()
    while (fitness > fitmin) and (trials <= maxtrials):
        optimizer = GlobalBestPSO(n_particles=Npart, dimensions=Nvar, options=psw_opts, bounds=bounds, ftol=ftol,
                                  ftol_iter=ftol_iter)
        fitness, pose = optimizer.optimize(fitnessFUN3Cam, iters=max_iter, verbose=True, **kwargs)
        trials += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(done after', trials, 'iterations in', round(elapsed_time, 2), 'seconds: fitness=', round(fitness, 3), ')')

    print('Computing rectified images by projective transformation: ', end='')
    start_time = time.time()
    ang = [pose[0], pose[1], pose[2]]
    pix = [pose[3], pose[3], 0]
    txy = [pose[4], pose[5]]
    Sr1 = np.concatenate((ang, pix, txy))
    Gr1 = proj.homography8dgf(Sr1)

    ang = [pose[6], pose[7], pose[8]]
    pix = [pose[9], pose[9], 0]
    txy = [pose[10], pose[11]]
    Sr2 = np.concatenate((ang, pix, txy))
    Gr2 = proj.homography8dgf(Sr2)

    ang = [pose[12], pose[13], pose[14]]
    pix = [pose[15], pose[15], 0]
    txy = [pose[16], pose[17]]
    Sr3 = np.concatenate((ang, pix, txy))
    Gr3 = proj.homography8dgf(Sr3)

    e_a = np.empty((0, 0))
    img1r = proj.proj_transform_numba(img1, Gr1, e_a, 'k')
    img2r = proj.proj_transform_numba(img2, Gr2, e_a, 'k')
    img3r = proj.proj_transform_numba(img3, Gr3, e_a, 'k')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(done in", round(elapsed_time, 2), "seconds)")

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(img1)
    axs[0, 1].imshow(img2)
    axs[0, 2].imshow(img3)
    axs[1, 0].imshow(img1r)
    axs[1, 1].imshow(img2r)
    axs[1, 2].imshow(img3r)
    fig.text(0.5, 0.90, 'Input unrectified images', ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.45, 'Output rectified images', ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.show()
def fourCam(**kwargs):
    print("Image rectification for four cameras")
    # Code for case 3
    from objective_function_4Cam import fitnessFUN4Cam
    ######################################################
    '''Parameters for the objective function and PSwO'''
    ftol = 1e-7         # tolerance of the fitness value for termination
    ftol_iter = 40      # number of iterations that fitness-value not improve more than 'ftol'
    max_iter = 2500     # maximum number of allowed iteration for PSwO algorithm
    psw_opts = {'c1': 0.5, 'c2': 0.3, 'w': 0.9} # coefficients of the PSwO algorithm
    fitmin = 1e-2       # minimum allowed fitness-value after PSwO termination
    maxtrials = 5       # maximum number of trials to obtain a solution fitness<=fitmin
    ######################################################
    Nvar = 6 * Ncam     # number of parameters to be optimized (six per camera)
    Npart = 12 * Nvar   # number of particles for the PSwO algorithm

    '''Reading stereo images to be rectified'''
    print('Reading stereo images to be rectified ', end='')
    start_time = time.time()
    img1, imgg1 = proj.load_images(dir + imname + '_1.png')
    img2, imgg2 = proj.load_images(dir + imname + '_2.png')
    img3, imgg3 = proj.load_images(dir + imname + '_3.png')
    img4, imgg4 = proj.load_images(dir + imname + '_4.png')
    MNc = img1.shape  # Shape of input images (rows,columns,channels)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(done in ", round(elapsed_time, 2), "seconds)")

    '''Detection of corresponding points in the input images (SIFT method)'''
    print('Detecting corresponding points using SIFT: ', end='')
    start_time = time.time()

    matchedPoints_c21, matchedPoints_c12, Np12 = proj.interest_points(imgg1, imgg2, Np0)
    matchedPoints_c31, matchedPoints_c13, Np13 = proj.interest_points(imgg1, imgg3, Np0)
    matchedPoints_c41, matchedPoints_c14, Np14 = proj.interest_points(imgg1, imgg4, Np0)
    pts_c12 = np.array([kp.pt for kp in matchedPoints_c12])  # List of coordinates of detected points in image 1
    pts_c21 = np.array([kp.pt for kp in matchedPoints_c21])  # List of coordinates of detected points in image 2
    pts_c13 = np.array([kp.pt for kp in matchedPoints_c13])  # List of coordinates of detected points in image 1
    pts_c31 = np.array([kp.pt for kp in matchedPoints_c31])  # List of coordinates of detected points in image 2
    pts_c14 = np.array([kp.pt for kp in matchedPoints_c14])  # List of coordinates of detected points in image 1
    pts_c41 = np.array([kp.pt for kp in matchedPoints_c41])  # List of coordinates of detected points in image 2
    Np = min([Np12,Np13,Np14])  # update the number of detected matched points
    pts_c12 = pts_c12[:Np, :]
    pts_c21 = pts_c21[:Np, :]
    pts_c13 = pts_c13[:Np, :]
    pts_c31 = pts_c31[:Np, :]
    pts_c14 = pts_c41[:Np, :]
    pts_c41 = pts_c14[:Np, :]
    # show_matched_features(img1, img2, matchedPoints_c12, matchedPoints_c21)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(detected ", Np, " points in ", round(elapsed_time, 2), "seconds)")

    '''Bounds of the search space for rectification parameters'''
    gamma_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    theta_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    phi_bnd = [-np.deg2rad(deg), np.deg2rad(deg)]
    f_bnd = [1 - scl, 1 + scl]  # Scaling parameter: (<1 reduce) (>1 increase) (=1 unscaling)
    tx_bnd = [-tx, tx]  # translation in x-direction (in normalized coordinates)
    ty_bnd = [-ty, ty]  # translation in y-direction (in normalized coordinates)

    # Specification of lower and upper bounds for particle swarm optimization
    lb = np.array([gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0],
                   gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0],
                   gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0],
                   gamma_bnd[0], theta_bnd[0], phi_bnd[0], f_bnd[0], tx_bnd[0], ty_bnd[0]])
    ub = np.array([gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1],
                   gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1],
                   gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1],
                   gamma_bnd[1], theta_bnd[1], phi_bnd[1], f_bnd[1], tx_bnd[1], ty_bnd[1]])

    '''Initialization of the Particle Swarm Optimization optimizer'''
    Kn = proj.gcoord(np.array(MNc))  # Transformation matrix Kn (3x3) for coordinate normalization
    bounds = (lb, ub)
    '''Additional parameters passed to the fitness function'''
    kwargs = {"pts_c12": pts_c12, "pts_c21": pts_c21, "pts_c13": pts_c13, "pts_c31": pts_c31,
              "pts_c14": pts_c14, "pts_c41": pts_c41, "Kn": Kn, "MNc": MNc, "w": w}
    '''Computation of projective transformation parameters for stereo rectification'''
    fitness = 1
    trials = 0
    print('Computing rectifying homographies: ', end='')
    start_time = time.time()
    while (fitness > fitmin) and (trials <= maxtrials):
        optimizer = GlobalBestPSO(n_particles=Npart, dimensions=Nvar, options=psw_opts, bounds=bounds, ftol=ftol,
                                  ftol_iter=ftol_iter)
        fitness, pose = optimizer.optimize(fitnessFUN4Cam, iters=max_iter, verbose=True, **kwargs)
        trials += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(done after', trials, 'iterations in', round(elapsed_time, 2), 'seconds: fitness=', round(fitness, 3), ')')

    print('Computing rectified images by projective transformation: ', end='')
    start_time = time.time()
    ang = [pose[0], pose[1], pose[2]]
    pix = [pose[3], pose[3], 0]
    txy = [pose[4], pose[5]]
    Sr1 = np.concatenate((ang, pix, txy))
    Gr1 = proj.homography8dgf(Sr1)

    ang = [pose[6], pose[7], pose[8]]
    pix = [pose[9], pose[9], 0]
    txy = [pose[10], pose[11]]
    Sr2 = np.concatenate((ang, pix, txy))
    Gr2 = proj.homography8dgf(Sr2)

    ang = [pose[12], pose[13], pose[14]]
    pix = [pose[15], pose[15], 0]
    txy = [pose[16], pose[17]]
    Sr3 = np.concatenate((ang, pix, txy))
    Gr3 = proj.homography8dgf(Sr3)

    ang = [pose[18], pose[19], pose[20]]
    pix = [pose[21], pose[21], 0]
    txy = [pose[22], pose[23]]
    Sr4 = np.concatenate((ang, pix, txy))
    Gr4 = proj.homography8dgf(Sr4)

    e_a = np.empty((0, 0))
    img1r = proj.proj_transform_numba(img1, Gr1, e_a, 'k')
    img2r = proj.proj_transform_numba(img2, Gr2, e_a, 'k')
    img3r = proj.proj_transform_numba(img3, Gr3, e_a, 'k')
    img4r = proj.proj_transform_numba(img4, Gr4, e_a, 'k')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("(done in", round(elapsed_time, 2), "seconds)")

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(img1)
    axs[0, 1].imshow(img2)
    axs[0, 2].imshow(img3)
    axs[0, 3].imshow(img4)
    axs[1, 0].imshow(img1r)
    axs[1, 1].imshow(img2r)
    axs[1, 2].imshow(img3r)
    axs[1, 3].imshow(img4r)
    fig.text(0.5, 0.83, 'Input unrectified images', ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.41, 'Output rectified images', ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.show()
############################################################################
'''Main program begins here!'''
############################################################################
'''Specification of configuration parameters'''
Ncam = 4            # Number of cameras of the multiocular vision system
dir = 'testimgs/'
imname = 'exp02'
Np0 = 250           # Number of desired corresponding points to be detected
deg = 60            # Specify the optimization bounds for rotation angles in the interval [-deg,deg]
scl = 0.3           # Specify the optimization bounds for scaling in the interval [1-scl,1+scl]
tx = 0.3            # Specify the optimization bounds for x-translation in the interval [-tx,tx]
ty = 0.3            # Specify the optimization bounds for y-translation in the interval [-ty,ty]
w = 0.95            # Optimization coefficient for trade-off between epipolar constraint and distortion minimization
###########################################################################
'''kwargs parameters for the switch-case'''
kwargs={"Ncam": Ncam, "Np0": Np0, "deg": deg, "scl": scl, "tx": tx, "w": w, "dir": dir, "imname": imname}
'''Define the switch-case dictionary with functions and kwargs'''
switch = {
    'twoCam': lambda **kwargs: twoCam(**kwargs),        # Pass kwargs to twoCam case
    'threeCam': lambda **kwargs: threeCam(**kwargs),    # Pass kwargs to threeCam case
    'fourCam': lambda **kwargs: fourCam(**kwargs),      # Pass kwargs to fourCam case
    'default': lambda **kwargs: twoCam(**kwargs)        # Pass kwargs to default (twoCam) case
}

# Switch-case logic
if Ncam == 2:
    argument = 'twoCam'
elif Ncam == 3:
    argument = 'threeCam'
else:
    argument = 'fourCam'
switch.get(argument, switch['default'])(**kwargs)
########################################################################################
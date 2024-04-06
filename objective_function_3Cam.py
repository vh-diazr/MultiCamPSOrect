import numpy as np
from numba import njit, prange
import projective_lib as proj

@njit(parallel=True)
def fitnessFUN3Cam(pose, pts_c12, pts_c21, pts_c13, pts_c31, Kn, MNc, w):
    """
    Calculate fitness value for multiocular stereo rectification.

    This function receives a pose array representing the projective parameters for each camera in the system,
    coordinates of corresponding points between cameras, transformation matrix from pixel to normalized coordinates,
    image size, and a weighting coefficient for optimization.

    Parameters:
        pose (array): (6*Ncam x 1) array with projective parameters of Ncam cameras.
        pts_cij (array): (Np x 2) array containing coordinates of Np corresponding points between cameras i and j.
        Kn (array): (3x3) transformation matrix from pixel coordinates to normalized coordinates within [-1,1;-1,1].
        MNc (array): (1x2) array containing the size [N,M] of the input images.
        w (float): Scalar weighting coefficient controlling the optimization trade-off.
                   Recommended value: w = 0.95.

    Returns:
        fitness (float): Scalar value representing fitness. Optimal value: fitness = 0.
    """

    # Container array (Swarm size x 1) for the fitness value for each particle.
    fitness = np.zeros(pose.shape[0])

    # Construction of a (4 x 2) array containing the corner points of the input images.
    ptsc = np.array([[0, 0], [0, MNc[0]], [MNc[1], 0], [MNc[1], MNc[0]]])

    # Reference the Fundamental Matrix of rectified images imposing an epipolar constraint.
    Fr = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0]], dtype=np.float64)

    Np = min([pts_c12.shape[0], pts_c21.shape[0], pts_c13.shape[0], pts_c31.shape[0]]) # Number of detected corresponding points.

    # Main loop for fitness evaluation of rectifying homographies
    for k in prange(pose.shape[0]):
        # Container arrays (3 x 3) for constructing the rectifying homographies.
        G1 = np.zeros((3, 3))
        G2 = np.zeros((3, 3))
        G3 = np.zeros((3, 3))

        # Construction of projective parameter arrays {S_c1, S_c2} and candidate rectifying homographies {G1, G2},
        # for each particle of the Swarm
        S_c1 = [pose[k, 0], pose[k, 1], pose[k, 2],
                pose[k, 3], pose[k, 3], 0,
                pose[k, 4], pose[k, 5]]
        G1[:, :] = proj.homography8dgf(S_c1)

        S_c2 = [pose[k, 6], pose[k, 7], pose[k, 8],
                pose[k, 9], pose[k, 9], 0,
                pose[k, 10], pose[k, 11]]
        G2[:, :] = proj.homography8dgf(S_c2)

        S_c3 = [pose[k, 12], pose[k, 13], pose[k, 14],
                pose[k, 15], pose[k, 15], 0,
                pose[k, 16], pose[k, 17]]
        G3[:, :] = proj.homography8dgf(S_c3)

        # Transformation of corresponding points to homogeneous coordinates
        ptsH_c12 = np.dot(Kn, proj.hom(pts_c12.T))
        ptsH_c21 = np.dot(Kn, proj.hom(pts_c21.T))
        ptsH_c13 = np.dot(Kn, proj.hom(pts_c13.T))
        ptsH_c31 = np.dot(Kn, proj.hom(pts_c31.T))
        # Transformation of image corner-points to homogeneous coordinates
        C = np.dot(Kn, proj.hom(ptsc.T))

        Pt21G2 = np.zeros(ptsH_c21.shape)
        Pt12G1 = np.zeros(ptsH_c12.shape)
        Pt31G3 = np.zeros(ptsH_c31.shape)
        Pt13G1 = np.zeros(ptsH_c13.shape)

        # Loop for evaluating the fitness of the epipolar constraint using rectifying homographies {G1, G2}.
        for i in range(ptsH_c12.shape[1]):
            Pt12G1[:, i] = np.dot(G1, np.ascontiguousarray(ptsH_c12[:, i]))
            Pt21G2[:, i] = np.dot(G2, np.ascontiguousarray(ptsH_c21[:, i]))
            Pt13G1[:, i] = np.dot(G1, np.ascontiguousarray(ptsH_c13[:, i]))
            Pt31G3[:, i] = np.dot(G3, np.ascontiguousarray(ptsH_c31[:, i]))
        FrPt21G2 = np.dot(Fr, Pt21G2)
        FrPt31G3 = np.dot(Fr, Pt31G3)
        J = np.abs(np.sum(Pt12G1 * FrPt21G2, axis=0)) + np.abs(np.sum(Pt13G1 * FrPt31G3, axis=0))
        j0 = np.mean(J) / 2 # Epipolar constraint fitness value

        # Projective transformation of image corner-points by rectifying homographies {G1, G2, G3}
        c1 = np.dot(G1, C)
        c2 = np.dot(G2, C)
        c3 = np.dot(G3, C)

        # Re-identification of image corner-points:
        #   Reference image: pA (upper-left), pB (downer-left), pC (downer-right), pD (upper-left)
        #   Rectified image by G1: pAt (upper-left), pBt (downer-left), pCt (downer-right), pDt (upper-left)
        pA, pB, pC, pD = C[:, 1], C[:, 0], C[:, 2], C[:, 3]
        pAt, pBt, pCt, pDt = c1[:, 1], c1[:, 0], c1[:, 2], c1[:, 3]


        # Computation of diagonal and horizontal lines from corner points in the reference image:
        #   lAC - line intersecting points AC;  lAD - line intersecting points AD;
        #   lBC - line intersecting points BC;  lBD - line intersecting points BD;
        lAC, lAD = np.cross(pA, pC), np.cross(pA, pD)
        lBC, lBD = np.cross(pB, pC), np.cross(pB, pD)

        # Computation of diagonal and horizontal lines from corner points in the rectified image by G1:
        #   lACt - line intersecting transformed points AC;  lAD - line intersecting transformed points AD;
        #   lBCt - line intersecting transformed points BC;  lBD - line intersecting transformed points BD;
        lACt, lADt = np.cross(pAt, pCt), np.cross(pAt, pDt)
        lBCt, lBDt = np.cross(pBt, pCt), np.cross(pBt, pDt)

        # Computation of intersecting points reference and transformed image lines:
        #   XAD - intersection point of lines {lAD,lADt}
        #   XBC - intersection point of lines {lBC,lBCt}
        XAD, XBC = np.cross(lAD, lADt), np.cross(lBC, lBCt)
        # j3 minimizes the scaling homogeneous coordinates of reference and transformed points {XAD,XBC}
        # (when j3 in minimized the lines {lAD,lADt} and {lBC,lBCt} by using G1 are parallel
        j3 = (np.abs(XAD[2]) + np.abs(XBC[2])) / 2

        # Computation of central points of reference and rectified images
        p0 = np.cross(lAC, lBD)     # Central point of reference image
        p1t = np.cross(lACt, lBDt)  # Central point of rectified image by G1

        # Re-identification of image corner-points of rectified image by G2:
        #   pAt (upper-left), pBt (downer-left), pCt (downer-right), pDt (upper-left)
        pAt, pBt, pCt, pDt = c2[:, 1], c2[:, 0], c2[:, 2], c2[:, 3]

        # Computation of diagonal and horizontal lines from corner points in the rectified image by G2:
        #   lACt - line intersecting transformed points AC;  lAD - line intersecting transformed points AD;
        #   lBCt - line intersecting transformed points BC;  lBD - line intersecting transformed points BD;
        lACt, lADt = np.cross(pAt, pCt), np.cross(pAt, pDt)
        lBCt, lBDt = np.cross(pBt, pCt), np.cross(pBt, pDt)

        # Computation of intersecting points reference and transformed image lines:
        #   XAD - intersection point of lines {lAD,lADt}
        #   XBC - intersection point of lines {lBC,lBCt}
        XAD, XBC = np.cross(lAD, lADt), np.cross(lBC, lBCt)

        # j4 minimizes the scaling homogeneous coordinates of reference and transformed points {XAD,XBC}
        # (when j4 in minimized the lines {lAD,lADt} and {lBC,lBCt} by using G2 are parallel
        j4 = (np.abs(XAD[2]) + np.abs(XBC[2])) / 2

        # Computation of central point of the rectified image by G2
        p2t = np.cross(lACt, lBDt)
        ##############################################################################################

        # Re-identification of image corner-points of rectified image by G3:
        #   pAt (upper-left), pBt (downer-left), pCt (downer-right), pDt (upper-left)
        pAt, pBt, pCt, pDt = c3[:, 1], c3[:, 0], c3[:, 2], c3[:, 3]

        # Computation of diagonal and horizontal lines from corner points in the rectified image by G2:
        #   lACt - line intersecting transformed points AC;  lAD - line intersecting transformed points AD;
        #   lBCt - line intersecting transformed points BC;  lBD - line intersecting transformed points BD;
        lACt, lADt = np.cross(pAt, pCt), np.cross(pAt, pDt)
        lBCt, lBDt = np.cross(pBt, pCt), np.cross(pBt, pDt)

        # Computation of intersecting points reference and transformed image lines:
        #   XAD - intersection point of lines {lAD,lADt}
        #   XBC - intersection point of lines {lBC,lBCt}
        XAD, XBC = np.cross(lAD, lADt), np.cross(lBC, lBCt)

        # j5 minimizes the scaling homogeneous coordinates of reference and transformed points {XAD,XBC}
        # (when j5 in minimized the lines {lAD,lADt} and {lBC,lBCt} by using G2 are parallel
        j5 = (np.abs(XAD[2]) + np.abs(XBC[2])) / 2

        # Computation of central point of the rectified image by G2
        p3t = np.cross(lACt, lBDt)

        # Computation of location error between central points of reference and rectified images
        #   j1 - location error for rectified image by G1
        #   j2 - location error for rectified image by G2
        j1 = np.sqrt(np.sum((p0 - p1t) ** 2))
        j2 = np.sqrt(np.sum((p0 - p2t) ** 2))
        j6 = np.sqrt(np.sum((p0 - p3t) ** 2))

        jminD = ((j1 + j3) / 2 + (j2 + j4) / 2 + (j5 + j6) / 2) / 3
        # Computation of final fitness value
        fitness[k] = w * j0 + (1 - w) * jminD

    return fitness

import cv2
import numpy as np
#import matplotlib.pylab as plt
from numba import jit, njit

def load_images(image_name):
    """
        Load and preprocess images.
        This function reads an image file, converts it from BGR to RGB color space,
        and then converts the RGB image to grayscale.
        Parameters:
            image_name (str): The filename of the image to load.
        Returns:
            img (numpy.ndarray): The loaded image in RGB color space.
            img_gray (numpy.ndarray): The grayscale version of the loaded image.
        Raises:
            FileNotFoundError: If the specified image file cannot be found.
        """
    # Read the image
    img = cv2.imread(image_name)
    # Check if the image was successfully loaded
    if img is None:
        raise FileNotFoundError(f"Unable to load image '{image_name}'")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert RGB to grayscale
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, imgg

def interest_points(img0g, imgg, Np):
    Npt = 4 * Np  # Numero de puntos totales
    Np0 = Npt  # Numero puntos para intentar asociar
    #imgg = cv2.normalize(cv2.convertScaleAbs(cv2.Laplacian(imgg, cv2.CV_64F)), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    #imgg = cv2.equalizeHist(imgg.astype(np.uint8))

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imgg, None)
    #img0g = cv2.normalize(cv2.convertScaleAbs(cv2.Laplacian(img0g, cv2.CV_64F)), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    #img0g = cv2.equalizeHist(img0g.astype(np.uint8))

    keypoints_0, descriptors_0 = sift.detectAndCompute(img0g, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors_0, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.25 * n.distance:
            good_matches.append(m)

    if len(good_matches) < Npt:
        Np0 = len(good_matches)

    mP1 = np.array([keypoints[m.queryIdx].pt for m in good_matches])
    mP2 = np.array([keypoints_0[m.trainIdx].pt for m in good_matches])

    index0 = []
    for k in range(Np0):
        dist = np.sqrt((mP1[k, 1] - mP2[k, 1]) ** 2)
        if dist < 0.1 * img0g.shape[1]:
            index0.append(k)
    index0 = np.array(index0)
    Np0 = len(index0)

    if len(index0) == 0:
        matchedPoints = []
        matchedPoints0 = []
    else:
        if Np0 < Np:
            Np = Np0

        index_e = np.random.choice(index0, size=Np, replace=False)

        matchedPoints = [keypoints[m.queryIdx] for m in [good_matches[i] for i in index_e]]
        matchedPoints0 = [keypoints_0[m.trainIdx] for m in [good_matches[i] for i in index_e]]

    return matchedPoints, matchedPoints0, Np

def show_matched_features(img1, img2, matchedPoints_c12, matchedPoints_c21):
    # Convert images to BGR format (if they are grayscale)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Convert OpenCV KeyPoint objects to numpy arrays
    pts1 = np.float32([kp.pt for kp in matchedPoints_c12])
    pts2 = np.float32([kp.pt for kp in matchedPoints_c21])

    # Draw matches
    matches = [cv2.DMatch(i, i, 0) for i in range(len(pts1))]
    matched_img = cv2.drawMatches(img1_color, matchedPoints_c12, img2_color, matchedPoints_c21, matches, None)

    plt.imshow(matched_img)
    plt.show()

    # Show the matched features
    #cv2.imshow("Matched Features", matched_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()






@njit
def gcoord(MN):
    if len(MN) == 1:
        MN = [MN[0], MN[0]]
    else:
        MN = [MN[0], MN[1]]

    Kn = np.array([[2, 0, -MN[1] - 1],
                   [0, 2, -MN[0] - 1],
                   [0, 0, max(MN) - 1]]) / (max(MN) - 1)
    return Kn
@njit
def rmat(ori):
    if len(ori) == 1:  # 2D Euler rotation matrix
        R = np.array([[np.cos(ori[0]), -np.sin(ori[0])],
                      [np.sin(ori[0]), np.cos(ori[0])]], dtype=np.float64)
    elif len(ori) == 3:  # 3D Euler rotation matrix
        R1 = rm(ori[0], 'z')
        R2 = rm(ori[1], 'y')
        R3 = rm(ori[2], 'z')
        R = np.dot(R3, np.dot(R2, R1))
    else:
        raise ValueError("Invalid input parameters.")
    return R
@njit
def rm(angle: float, xyz: str) -> np.ndarray:
    '''
    Return the 3D rotation matrix by the given angle around the 'x' or 'y' or 'z' axis.
    Parameters:
        angle (float): Angle of rotation in radians.
        xyz (str): Axis of rotation ('x', 'y', or 'z').
    Returns:
        R (numpy.ndarray): 3x3 rotation matrix.
    '''
    s = np.sin(angle)
    c = np.cos(angle)

    if xyz == 'x':
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, c, -s],
                      [0.0, s, c]], dtype=np.float64)
    elif xyz == 'y':
        R = np.array([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]], dtype=np.float64)
    elif xyz == 'z':
        R = np.array([[c, -s, 0.0],
                      [s, c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError("Invalid axis, must be 'x', 'y', or 'z'.")
    return R

@njit
def homography8dgf(T):
    '''
    Compute a 3x3 homography matrix.
    Parameters:
        T (list or array-like): 8x1 vector of parameters.
    Returns:
        G (numpy.ndarray): Homography matrix.
    '''
    Q = rmat(T[:3])
    K = np.array([[T[3], T[5], T[6]],
                  [0, T[4], T[7]],
                  [0, 0, 1.]], dtype=np.float64)
    G = np.dot(K, np.hstack((Q[:, :2], np.array([[0], [0], [1]]))))
    return G

@njit
def hom(Y, S=1):
    """
    Return the homogeneous coordinates (Hs) of Y with scale S.
    Parameters:
        Y (numpy.ndarray): Matrix of coordinates.
        S (float): Scale factor (default is 1).
    Returns:
        Hs (numpy.ndarray): Homogeneous coordinates with scale S.
    """
    # if S is None:
    #    S = 1
    ones_row = np.ones((1, Y.shape[1])) * S
    Hs = np.vstack((Y, ones_row))
    return Hs
# @njit
# def proj_transform(I, ang, t, bc='k'):
#     MN = I.shape
#
#     roiA = [1, 1, I.shape[1], I.shape[0]]
#     Xa, Ya, xa, ya = gcoord_proj(MN)
#     Xb, Yb, xb, yb = Xa, Ya, xa, ya
#
#     if bc == 'k':
#         fc = np.zeros((1, 1, I.shape[2]))
#     elif bc == 'w':
#         fc = np.ones((1, 1, I.shape[2])) * 255
#     elif bc == 'a':
#         fc = np.mean(I, axis=(0, 1))
#
#     if ang.size == 9:
#         G = ang
#     else:
#         R = rmat(ang + np.pi * np.array([1, 0, 1]))
#         G = np.hstack((R[:, :2], t.reshape(-1, 1)))
#
#     II = np.linalg.solve(G, np.vstack((Xb.ravel(), Yb.ravel(), np.ones_like(Xa.ravel()))))
#     Ix = (II[0] / II[2]).reshape(Xa.shape)
#     Iy = (II[1] / II[2]).reshape(Xa.shape)
#
#     IDx, IDy = gcoord_idx(Ix, Iy, MN)
#     points = np.column_stack((IDx.ravel(), IDy.ravel()))
#
#     J = np.zeros_like(I)
#     for i in range(MN[0]):
#         for j in range(MN[1]):
#             if IDx[i, j] < roiA[0] or IDx[i, j] > roiA[2] or \
#                     IDy[i, j] < roiA[1] or IDy[i, j] > roiA[3]:
#                 J[i, j, :] = fc
#             else:
#                 xe = IDx[i, j] - roiA[0] + 1
#                 ye = IDy[i, j] - roiA[1] + 1
#                 H = I[int(np.floor(ye)):int(np.ceil(ye)), int(np.floor(xe)):int(np.ceil(xe)), :]
#
#                 J[i, j, :] = aux_interp(xe - np.floor(xe) + 1, ye - np.floor(ye) + 1, H)
#     return J

@jit(nopython=True)
def proj_transform_numba(I, ang, t, bc='k'):
    MN = I.shape

    roiA = [1, 1, I.shape[1], I.shape[0]]
    Xa, Ya, xa, ya = gcoord_proj(MN)
    Xb, Yb, xb, yb = Xa, Ya, xa, ya

    if bc == 'k':
        fc = np.zeros((1, 1, I.shape[2]), dtype=np.float64)
    elif bc == 'w':
        fc = np.ones((1, 1, I.shape[2]), dtype=np.float64) * 255
    elif bc == 'a':
        fc = np.ones((1, 1, I.shape[2]), dtype=np.float64) * np.mean(I)

    if ang.size == 9:
        G = ang
    else:
        R = rmat(np.ravel(ang) + np.pi * np.array([1, 0, 1]))
        G = np.hstack((R[:, :2], t.reshape(-1, 1)))

    II = np.empty((3, Xb.size), dtype=np.float64)
    II[0, :], II[1, :], II[2, :] = np.linalg.solve(G, np.vstack((Xb.ravel(), Yb.ravel(), np.ones_like(Xa.ravel()))))
    Ix = (II[0, :] / II[2, :]).reshape(Xa.shape)
    Iy = (II[1, :] / II[2, :]).reshape(Xa.shape)

    IDx, IDy = gcoord_idx(Ix, Iy, MN)
    points = np.column_stack((IDx.ravel(), IDy.ravel()))

    J = np.zeros_like(I, dtype=np.float64)
    for i in range(MN[0]):
        for j in range(MN[1]):
            if IDx[i, j] < roiA[0] or IDx[i, j] > roiA[2] or \
                    IDy[i, j] < roiA[1] or IDy[i, j] > roiA[3]:
                J[i, j, :] = fc.ravel()
            else:
                xe = IDx[i, j] - roiA[0] + 1
                ye = IDy[i, j] - roiA[1] + 1
                H = I[int(np.floor(ye)):int(np.ceil(ye)), int(np.floor(xe)):int(np.ceil(xe)), :]

                J[i, j, :] = aux_interp(xe - np.floor(xe) + 1, ye - np.floor(ye) + 1, H)
    return J.astype(np.uint8)

@njit
def aux_interp(xe, ye, I):
    MN = I.shape
    J = np.zeros_like(I)  # Initialize J with zeros, matching the shape and dtype of I
    if MN[0] == 1 and MN[1] == 1:
        return I.copy()  # Return a copy of I
    elif MN[0] == 2 and MN[1] == 1:
        J[0, 0, :] = I[0, 0, :] + (ye - int(ye)) * (I[1, 0, :] - I[0, 0, :])
    elif MN[0] == 1 and MN[1] == 2:
        J[0, 0, :] = I[0, 0, :] + (xe - int(xe)) * (I[0, 1, :] - I[0, 0, :])
    else:
        X = I[:, 0, :] + (xe - int(xe)) * (I[:, 1, :] - I[:, 0, :])
        J[0, :] = X[0, :] + (ye - int(ye)) * (X[1, :] - X[0, :])
    return J
@njit
def gcoord_proj(MN):
    if isinstance(MN, int):
        MN = [MN, MN]

    bx = np.empty(MN[1], dtype=np.int64)
    by = np.empty(MN[0], dtype=np.int64)

    max_val = 0
    for i in range(len(MN)):
        if MN[i] > max_val:
            max_val = MN[i]

    for i in range(MN[1]):
        bx[i] = 2 * (i - MN[1] // 2)

    for i in range(MN[0]):
        by[i] = 2 * (i - MN[0] // 2)

    mp = max_val - 1
    x = bx / mp
    y = by / mp

    # Create X and Y manually without using numpy.meshgrid
    X = np.empty((len(y), len(x)))
    Y = np.empty((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            X[i, j] = x[j]
            Y[i, j] = y[i]

    return X, Y, x, y
@njit
def gcoord_idx(x0, y0, MN):
    if isinstance(MN, int):
        MN = [MN, MN]
    mp = max(MN) - 1
    ix = (mp * x0 + MN[1] - 1) / 2 + 1
    iy = (mp * y0 + MN[0] - 1) / 2 + 1
    return ix, iy
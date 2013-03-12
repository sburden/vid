
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2013 

import numpy as np
import pylab as plt

def pinh(dh,A):
    """
    px = pinh  apply pinhole camera model to (distorted) homogeneous
               camera coordinates

    INPUTS
      dh - 3 x N - (distorted) homogeneous camera coordinates
      A - 3 x 3 - pinhole camera matrix

    OUTPUTS
      px - 2 x N - pixels in camera field
    """
    px = np.dot(A, dh / dh[(2,2,2),:])

    return px[0:2,:] / px[(2,2),:]

def dist(uh,d):
    """
    dh = dist  distort homogeneous coordinates with radial/tangential model

    INPUTS
      uh - 3 x N - points in undistorted homogeneous camera coordinates
      d - 1 x 4 - distortion coefficients
        - [ d[0] * r^2, d[1] * r^4, d[2] * x*y, d[3] * r^2+2x^2 ]

    OUTPUTS
      dh - 3 x N - points in distorted homoegeous camera coordinates
    """
    N = uh.shape[1]
    d = d.flatten()
    xp = uh[0,:] / uh[2,:]
    yp = uh[1,:] / uh[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    xpp = xp*(1+d[0]*r2+d[1]*r4) + 2*d[2]*xp*yp + d[3]*(r2+2*xp**2)
    ypp = yp*(1+d[0]*r2+d[1]*r4) + d[2]*(r2+2*yp**2) + 2*d[3]*xp*yp 
    dh = np.vstack((xpp,ypp,np.ones((1,N))))

    return dh

def rigid(p,R,t):
    """
    q = rigid  apply rigid transformation to points

    INPUTS
      p - 3 x N - 3D points to transform
      R - 3 x 3 - rotation matrix
      t - 3 x 1 - translation vector

    OUTPUTS
      q - 3 x N - transformed 3D points
    """
    N = p.shape[1]
    q = np.dot(R, p) + np.kron(np.ones((1,N)), t)

    return q

def dlt(p,dlt):
    """
    q = dlt  apply direct linear transformation

    INPUTS
      p - 3 x N - 3D locations
      dlt - 3 x 4 - direct linear transformation

    OUTPUTS
      q - 2 x N - pixel locations
    """
    N = p.shape[1]
    q = np.dot(dlt, np.vstack((p, np.ones((1,N)))))

    return q[0:2,:] / q[(2,2),:]

def zhang(p, R, t, A, d=np.zeros((1,4))):
    """
    q = zhang  transform points using the Zhang camera model

    A good review/comparison of camera calibration techniques:
        W. Suna nd J.R. Cooperstock
        An empirical evaluation of factors influencing camera calibration
        Machine Vision and Applications
        No 1, Vol 17, Pg 51-67, 2006
        DOI 10.1007/s00138-006-0014-6

    INPUTS
      p - 3 x N - 3D points
      R - 3 x 3 - rotation matrix
      t - 3 x 1 - translation vector
      A - 3 x 3 - pinhole camera matrix
      d - 1 x 4 - (optional) distortion coefficients

    OUTPUTS
      q - 2 x N - pixel locations
    """
    q = pinh( dist( rigid(p, R, t), d), A)

    return q


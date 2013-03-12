
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
# (c) Sam Burden, UC Berkeley, 2010 
# (c) Shai Revzen, U Penn, 2010 

import sys
import os

import numpy as np
import scipy as sp
import scipy.ndimage as si
import pylab as plt

try:
    import cv
except ImportError:
    cv = None

from util.cvadaptors import *
from util import Struct, Lambda

class Base(object):
    def __init__(self):
        """
        src.Base  abstract base class for framesource plugins

        Provides null implementations of required functions.

        .ini  struct containing initialization configuration
        .cfg  configuration generated from ini data
        .imgs  array of images
            .imgs[k].im -- actual image
            .imgs[k].id -- id used to retrieve the image
            .imgs[k][*] -- various image-specific metadata
        .new()  initialize from .ini
        .get()  obtain an image
        .op()  process an image
        .argf  arguments propagating forward through framesource
        .argr  arguments propagating backward through framesource
        .xf()  transform .argf forward through framesource
        .xr()  transform .argr backward through framesource

        """
        self.ini  = Struct() 
        self.cfg  = Struct() 
        self.imgs = [] 
        self.argf = Struct()  
        self.argr = Struct()  

    def new(self, ini=None):
        """
        src.new(ini)  initialize framesource
        """
        pass

    def get(self, ids, srcs=None):
        """
        img = src.get(ids, srcs)  obtain images specified by ids
        """
        return None

    def op(self, img):
        """
        img = src.op(img)  process an image
        """
        return img.copy()
    
    def xf(self, argf=None):
        """
        argf = src.xf(argf)  transform argf forward through framesource
        src.xf()  transform src.argf forward through framesource
        """
        return None

    def xr(self, argr=None):
        """
        argr = src.xr(argr)  transform argr backward through framesource
        src.xr()  transform src.argr backward through framesource
        """
        return None

class CVAvi(Base):
    def __init__(self, pth):
        """
        avi = src.CVAvi(pth)  creates framesource from .avi file
          using the OpenCV video I/O interface

        INPUTS
          pth - string - path to .avi file

        OUTPUTS
          avi - framesource - avi framesource
        """
        # Obligatory call to base class
        Base.__init__(self)

        # Store path to file
        self.ini.pth = pth

        # Initialize fields
        self.N = 0

        # Initialize framesource
        self.new()

    def __repr__(self):
        return ('src.CVAvi("'+self.ini.pth+'")')

    def new(self, ini=None):
        if not ini:
            ini = self.ini

        self.cfg.cap = cv.CaptureFromFile(self.ini.pth)
        self.N = int(cv.GetCaptureProperty(self.cfg.cap, 
                                           cv.CV_CAP_PROP_FRAME_COUNT))

        img = self.get([1])[0].im

        self.sz = (int(cv.GetCaptureProperty(self.cfg.cap,
                                             cv.CV_CAP_PROP_FRAME_HEIGHT)),
                   int(cv.GetCaptureProperty(self.cfg.cap,
                                             cv.CV_CAP_PROP_FRAME_WIDTH)), 
                   img.shape[2])

    def get(self, ids, srcs=None):
        self.imgs = []
        for id in ids:
            if 0 <= id < self.N:
                cv.SetCaptureProperty(self.cfg.cap, 
                                      cv.CV_CAP_PROP_POS_FRAMES, id)

                img = Struct(im  = cv2array(cv.QueryFrame(self.cfg.cap)),
                             id  = id,
                             pth = self.ini.pth)

                self.imgs.append(img)

            else:
                print "WARNING  id #%d not between 0 and %d" % (id, self.N-1)

        return self.imgs

if cv is None:
    class CVAvi(Base):
        def __init__(self,pth):
            raise ImportError,"OpenCV not available for import"

class Op(Base):
    def __init__(self, op, arg=()):
        """
        opsrc = src.Op(op, args)  executes the given operation on frames

        INPUTS
          op - func - operation acting on whole image
          arg - tuple - extra arguments to pass to op
        """
        # Obligatory call to base class
        Base.__init__(self)

        # Store function
        if isinstance(op,str):
            self.ini.op    = Lambda(op)
        else:
            self.ini.op = op
        self.ini.arg = arg

    def __repr__(self):
        return ('src.Op('+str(self.ini.op)+','+str(self.ini.arg)+')')

    def new(self, ini=None):
        Op.__init__(self, self.ini.op, self.ini.arg)

    def op(self, img):
        jmg = img.copy()

        jmg.im = self.ini.op(img.im, *self.ini.arg)

        return jmg

class Lum(Base):
    def __init__(self, ord=1, sbs=1):
        """
        lum = src.Lum  correct luminosity by polynomial fitting

        INPUTS
          ord - int - order of polynomial to use
          sbs - int - subsampling step size

        By Shai Revzen, Berkeley 2006
        """
        Base.__init__(self)

        self.ini.sz  = None
        self.ini.ord = ord
        self.ini.sbs = sbs

    def __repr__(self):
        return ('src.Lum('+str(self.ini.ord)+','+str(self.ini.sbs)+')')

    def allocate(self, sz):
        X = np.ones(sz[0:2])
        Y = np.ones(sz[0:2])
        xyc = np.ones((X.size,self.ini.ord*2+1))
        for o in range(self.ini.ord):
            X = np.cumsum(X,axis=1)
            Y = np.cumsum(Y,axis=0)
            xyc[:,2*(o+1)-1] = X.flatten(1)
            xyc[:,2*(o+1)]   = Y.flatten(1)
        self.cfg.xyc = xyc
        self.cfg.xyc0 = xyc[::self.ini.sbs,:]

    def op(self, img):
        jmg = img.copy()

        if not self.ini.sz:
            self.allocate(img.im.shape)

        if len(jmg.im.shape) < 3:
            jmg.im = jmg.im.reshape(list(jmg.im.shape)+[1])
        
        for k in range(jmg.im.shape[2]):
            im  = jmg.im[...,k]
            sz  = im.shape
            im  = im.flatten(1)
            smp = im[::self.ini.sbs]
            bg  = smp.min()
            fg  = np.logical_not(smp == bg)
            cof = np.linalg.lstsq(self.cfg.xyc0[fg,:],smp[fg])[0]
            im  = im - np.dot(self.cfg.xyc, cof)
            im[im==bg] = np.median(im[fg])
            jmg.im[...,k] = im.reshape(sz, order='f')

        jmg.im = (jmg.im - jmg.im.min()) / (jmg.im.max() - jmg.im.min()) 

        if jmg.im.shape[2] == 1:
            jmg.im = jmg.im[...,0]

        return jmg

class Saturate(Base):
    def __init__(self, *args):
        """
        sat = src.Saturate([low,] high)  saturates luminosity

        INPUTS
          high - scalar - largest luminance allowed
          (optional)
          low - scalar - smallest luminance allowed
        """
        # Obligatory call to base class
        Base.__init__(self)

        if len(args) == 1:
            self.ini.low = 0
            self.ini.high = args[0]
        else:
            self.ini.low = args[0]
            self.ini.high = args[1]

    def __repr__(self):
        return ('src.Saturate('+str(self.ini.low)+','
                +str(self.ini.high)+')')

    def op(self, img):
        jmg = img.copy()

        jmg.im[jmg.im < self.ini.low] = self.ini.low
        jmg.im[jmg.im > self.ini.high] = self.ini.high

        return jmg

class Resample(Base):
    def __init__(self, tbl):
        """
        res = src.Resample(tbl)  resample frames from a source

        INPUTS
          tbl - list - table of frame indices
        """
        Base.__init__(self)

        self.ini.tbl = tbl
        self.N = len(tbl)

    def __repr__(self):
        return ('src.Resample('+str(self.ini.tbl)+')')

    def new(self, ini=None):
        Resample.__init__(self, self.ini.tbl)

    def get(self, ids, srcs):
        # Split sources into raw and resampler
        raw = srcs[0:-1]
        res = srcs[-1]

        # Translate frame requests
        jds = [res.ini.tbl[i] for i in ids]

        reqPipe(raw, jds)
        opPipe(raw)

        self.imgs = []
        for k,img in enumerate(raw[-1].imgs):
            jmg = img.copy()

            if not hasattr(jmg,'id0'):
               jmg.id0 = [] 

            jmg.id0.append(img.id)
            jmg.id = ids[k]

            self.imgs.append(jmg)

        return self.imgs

class Shift(Base):
    def __init__(self, dp):
        """
        sh = src.Shift  shift framesource

        INPUTS
          dp - 2-tuple - number of pixels to shift frame
        """
        Base.__init__(self)

        self.ini.dp = dp
        self.ini.xf = lambda x : x + np.kron(np.ones((1,x.shape[1])), np.array([[dp[1],dp[0]]]).T)
        self.ini.xr = lambda x : x - np.kron(np.ones((1,x.shape[1])), np.array([[dp[1],dp[0]]]).T)

    def new(self, ini=None):
        Transform.__init__(self, self.ini.xf, self.ini.xr)

    def op(self, img):
        jmg = img.copy()

        d = jmg.im.shape[2]
        sz = jmg.im.shape[0:2]
        dp = self.ini.dp
        jmg.im = np.vstack((np.zeros((dp[0],dp[1]+sz[1],d)),
                                np.hstack((np.zeros((sz[0],dp[1],d)),
                                           jmg.im)) ))

        return jmg

    def xf(self, argf={}):
        pts = None
        if 'pts' in argf.keys():
            pts = self.ini.xf(argf.pts)
        return pts

    def xr(self, argr={}):
        pts = None
        if 'pts' in argr.keys():
            pts = self.ini.xr(argr.pts)
        return pts

class Hstack(Base):
    def __init__(self, srcs):
        """
        hs = src.Hstack  stack framesources horizontally

        INPUTS
          srcs - list - framesource pipe / plugin chain
        """
        Base.__init__(self)

        self.cfg.srcs = srcs

        self.N   = np.array([info(sc).N for sc in srcs],  dtype=int).min()
        self.szs = np.array([info(sc).sz for sc in srcs], dtype=int)
        if any(self.szs[:,0] - self.szs[0,0]):
            raise IndexError, "Framesources do not share a common height"
        self.sz  = [self.szs[0,0], self.szs[:,1].sum(), self.szs[0,2]]

    def __repr__(self):
        return ('src.Hstack('+str(srcs)+')')

    def get(self, ids, srcs=None):
        self.imgs = []
        for id in ids:
            if 0 <= id < self.N:
                ims = [getIm(sc,id) for sc in self.cfg.srcs]

                img = Struct(im  = np.hstack(tuple(ims)),
                             id  = id,
                             pth = 'src.Hstack')

                self.imgs.append(img)

            else:
                print "WARNING  id #%d not between 0 and %d" % (id, self.N-1)

class Vstack(Base):
    def __init__(self, srcs):
        """
        vs = src.Vstack  stack framesources vertically

        INPUTS
          srcs - list - framesource pipe / plugin chain
        """
        Base.__init__(self)

        self.cfg.srcs = srcs

        self.N   = np.array([info(sc).N for sc in srcs],  dtype=int).min()
        self.szs = np.array([info(sc).sz for sc in srcs], dtype=int)
        if any(self.szs[:,1] - self.szs[0,1]):
            raise IndexError, "Framesources do not share a common width"
        self.sz  = [self.szs[:,0].sum(), self.szs[0,1], self.szs[0,2]]

    def __repr__(self):
        return ('src.Vstack('+str(srcs)+')')

    def get(self, ids, srcs=None):
        self.imgs = []
        for id in ids:
            if 0 <= id < self.N:
                ims = [getIm(sc,id) for sc in self.cfg.srcs]

                img = Struct(im  = np.vstack(tuple(ims)),
                             id  = id,
                             pth = 'src.Vstack')

                self.imgs.append(img)

            else:
                print "WARNING  id #%d not between 0 and %d" % (id, self.N-1)

class Transform(Base):
    def __init__(self, xf, xr):
        """
        tfm = src.Transform  transform framesource

        INPUTS
          xf - func - diffeomorphism of the plane
          xr - func - inverse of xf:  xf(xr) = id
        """
        Base.__init__(self)

        self.ini.xf = xf
        self.ini.xr = xr

    def new(self, ini=None):
        Transform.__init__(self, self.ini.xf, self.ini.xr)

    def op(self, img):
        jmg = img.copy()

        # Apply transformation to corners to guess bounding box
        sz = jmg.im.shape
        c = np.array([[0,0,sz[0],sz[0]],[0,sz[1],0,sz[1]]])
        d = self.ini.xf(c)
        bx = tuple(d.max(1))

        # Apply transformation to each channel
        for k in range(jmg.im.shape[2]):
            jmg.im[...,k] = si.geometric_transform(jmg.im[...,k],
                                                       self.ini.xf,
                                                       output_shape=bx)

        return jmg

    def xf(self, argf={}):
        pts = None
        if 'pts' in argf.keys():
            pts = self.ini.xf(argf.pts)
        return pts

    def xr(self, argr={}):
        pts = None
        if 'pts' in argr.keys():
            pts = self.ini.xr(argr.pts)
        return pts

class ROI(Base):
    def __init__(self, roi):
        """
        roi = src.ROI  region-of-interest framesource

        INPUTS
          roi - Nr x Nc - pixel mask
              - (r,c) - arguments to np.ix_
        """
        Base.__init__(self)

        self.ini.roi = roi
        if isinstance(roi,tuple) or isinstance(roi,list):
            self.cfg.r = np.array(roi[0])
            self.cfg.c = np.array(roi[1])
        else:
            self.cfg.r   = np.any(roi,axis=1).nonzero()
            self.cfg.c   = np.any(roi,axis=0).nonzero()

        self.sz = (self.cfg.r.shape[0],self.cfg.c.shape[0],1)

    def __repr__(self):
        if isinstance(self.ini.roi,tuple) or isinstance(self.ini.roi,list):
            roi = str(self.ini.roi)
        else:
            roi = 'np.'+str(self.ini.roi)
        return ('src.ROI('+roi+')')

    def new(self, ini=None):
        ROI.__init__(self, self.ini.roi)

    def op(self, img):
        jmg = img.copy()

        jmg.im = img.im[np.ix_(self.cfg.r,self.cfg.c)]

        return jmg

class AosRaw(Base):
    def __init__(self, pth, sz=(600,800,1), frpad=40., hlen=244.+800, xraw=False):
        """
        avi = src.AosRaw  creates framesource from .raw file
          by reading the format directly from binary

        INPUTS
          pth - string - path to .raw file
              - if file ends with .xraw or xraw=True is set, 
                uses XRAW file format

        OUTPUTS
          raw - framesource - raw framesource
        """
        # Obligatory call to base class
        Base.__init__(self)

        # Store path to file
        self.ini.pth = pth
        self.ini.sz = sz
        self.ini.frpad = frpad
        self.ini.hlen = hlen
        if xraw or pth.endswith('xraw'):
            self.ini.xraw = True
        else:
            self.ini.xraw = False

        # Initialize fields
        self.N = 0

        # Initialize framesource
        self.new()

    def __repr__(self):
        return ('src.AosRaw("'+self.ini.pth+'",'+str(self.ini.sz)
                 +','+str(self.ini.frpad)+','+str(self.ini.hlen)
                 +','+str(self.ini.xraw)+')')

    def new(self, ini=None):
        if not ini:
            ini = self.ini

        self.sz        = self.ini.sz
        self.cfg.fname = self.ini.pth
        self.cfg.flen  = os.path.getsize(self.ini.pth)
        self.cfg.frpad = self.ini.frpad
        self.cfg.hlen  = self.ini.hlen
        self.cfg.frlen = np.array(self.sz).prod()
        self.N         = ((self.cfg.flen-self.cfg.hlen)
                           / (self.cfg.frlen+self.cfg.frpad) - 1)
        if not self.N == np.floor(self.N):
            print 'src.AosRaw:  file length is not an integer number of frames'
            self.N = np.floor(self.N)

    def get(self, ids, srcs=None):
        self.imgs = []
        for id in ids:
            if 0 <= id < self.N:
                fi = open(self.ini.pth,'r')
                fi.seek(id*(self.cfg.frlen + self.cfg.frpad) 
                        + self.cfg.hlen)
                #im = io.fread(fi, self.cfg.frlen, 'b').reshape(self.sz)
                im = np.fromfile(fi, 
                                 count=self.cfg.frlen, 
                                 dtype=np.uint8).reshape(self.sz)
                fi.close()

                #ind = np.array((np.kron(np.ones((1,self.sz[1]/4.)), [4,3,2,1])
                #       + np.kron(np.ones((4,1)),np.arange(0,self.sz[1],4)).reshape(1,self.sz[1])),dtype=int)

                if self.ini.xraw:
                    ind = np.array((np.kron(np.ones((self.sz[1]/4.)), [3,2,1,0])
                          + np.kron(np.ones((4,1)),np.arange(0,self.sz[1],4)).flatten('f')),dtype=int)
                else:
                    ind = np.arange(0,self.sz[1])

                img = Struct(im  = np.array(im[:,ind][...,0],dtype=np.uint8),
                             id  = id,
                             pth = self.ini.pth)

                self.imgs.append(img)

            else:
                print "WARNING  id #%d not between 0 and %d" % (id, self.N-1)

        return self.imgs

class Pth(Base):
    def __init__(self, pth, **args):
        """
        pth = src.Pth  creates framesource from multiple files in path

        INPUTS
          pth - string - path to expand with glob
            e.g. 'raw/Batch_0960*.raw'

        OUTPUTS
          pth - framesource - path framesource
        """
        # Obligatory call to base class
        Base.__init__(self)

        # Store path to file
        self.ini.pth = pth
        self.ini.args = args

        # Initialize fields
        self.N = 0

        # Initialize framesource
        self.new()

    def __repr__(self):
        return ('src.Pth("'+self.ini.pth+'",**'+str(self.ini.args)+')')

    def new(self, ini=None):
        if not ini:
            ini = self.ini

        from glob import glob

        self.pth = ini.pth
        self.fis = glob(self.pth)
        self.N   = len(self.fis)
        self.sz  = list(getIm([self],0).shape)

    def get(self, ids, srcs=None):
        self.imgs = []
        for id in ids:
            if 0 <= id < self.N:

                fi = self.fis[id]

                if fi.endswith('raw'):
                    xraw = False
                    args = self.ini.args
                    if 'xraw' in args.keys():
                        xraw = args['xraw']
                    im = getIm([AosRaw(fi,xraw=xraw)],0)
                else:
                    im = cv2array(cv.LoadImage(fi))
    
                img = Struct(im = im, id = id, pth = fi)

                self.imgs.append(img)

            else:
                print "WARNING  id #%d not between 0 and %d" % (id, self.N-1)

        return self.imgs

class Imgs(Base):
    def __init__(self, imgs):
        """
        imgs = src.Imgs  creates framesource from list of imgs

        INPUTS
          imgs - list - imgs[i] is an image

        OUTPUTS
          imgs - framesource - imgs framesource
        """
        Base.__init__(self)

        self.ini.imgs = imgs

        self.N = 0

        self.new()

    def new(self, ini=None):
        if not ini:
            ini = self.ini

        self.N   = len(self.ini.imgs)
        self.sz  = list(self.ini.imgs[0].shape)

    def get(self, ids, srcs=None):
        self.imgs = []
        for id in ids:
            if 0 <= id < self.N:

                img = Struct(im = self.ini.imgs[id], id = id, pth = 'src.Imgs')

                self.imgs.append(img)

            else:
                print "WARNING  id #%d not between 0 and %d" % (id, self.N-1)

        return self.imgs

def reqPipe(srcs, ids):
    """
    src.reqPipe(srcs, ids)  request images in the framesource pipe

    INPUTS
      srcs - list - framesource pipe / plugin chain
      ids - list - requested frame ID #'s

    MODIFIES
      srcs  to update .imgs fields with processed frames
    """
    # Find the first `raw' frame source
    for k in reversed(range(len(srcs))):
        imgs = srcs[k].get(ids, srcs[0:k+1])
        if imgs:
            break

    # Record the request in the source that processed it
    srcs[k].cfg.ids = ids

    # Run the image processing pipe on what was returned
    opPipe(srcs[k:])

def opPipe(srcs):
    """
    src.opPipe(srcs)  process images in the framesource pipe

    INPUTS
      srcs - list - framesource pipe / plugin chain

    MODIFIES
      srcs  to update .imgs fields with processed frames
    """
    for k in range(1,len(srcs)):
        srcs[k].imgs = []
        for img in srcs[k-1].imgs:
            srcs[k].imgs.append(srcs[k].op(img))

def info(srcs, keys=['N','sz']):
    """
    info = src.info(srcs)  collect source info

    INPUTS
      srcs - list - framesource pipe / plugin chain
      keys - list - keys to collect; default is ['N','sz']

    OUTPUT
      info - struct -requested info
    """
    info = Struct()
    for sc in srcs:
        for key in keys:
            if hasattr(sc, key):
                info.__dict__[key] = getattr(sc,key);

    if 'sz' in keys and len(info.sz) == 2:
       info.sz.append(1) 

    return info

def apply(srcs, imgs):
    """
    jmgs = src.apply  apply a framesource to images

    INPUTS
      srcs - list - framesource pipe / plugin chain
      imgs - list - images to process

    OUTPUT
      jmgs - list - processed images
    """
    jmgs = []
    imsc = [Imgs(imgs)]+srcs
    for k in range(len(imgs)):
        jmgs.append(getIm(imsc,k))

    return jmgs

def getImg(srcs, id):
    """
    img = src.getImg  obtains image object from the framesource pipe

    INPUTS
        srcs - list - framesource pipe / plugin chain
        id - int - requested frame ID #

    OUTPUT
        img - struct -image #id from pipe
           .im - np.array - image array
           .pth - string - image path
           .id - int - image id#
    """
    # Request images in pipe
    reqPipe(srcs, [id])

    # Process images in pipe
    opPipe(srcs)
    
    # Return processed image
    return srcs[-1].imgs[0]

def getIm(srcs, id):
    """
    im = src.getIm  obtains image array from the framesource pipe

    INPUTS
        srcs - list - framesource pipe / plugin chain
        id - int - requested frame ID #

    OUTPUT
        im - np.array - image #id from pipe
    """
    return getImg(srcs,id).im

def pager(srcs):
    """
    src.pager()  interface for displaying and interacting w/ srcs

    INPUTS
      srcs - list - framesource pipe / plugin chain
    """
    import wx 

    import cv 

    import src

    import numpy as np
    import scipy as sp
    import scipy.ndimage as si
    import pylab as plt
    import sys

    pts = []

    class myFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, -1, 'Framesource playback')

            self.pnl = wx.Panel(self)

            self.Bind(wx.EVT_IDLE, self.onIdle)

            self.pnl.Bind(wx.EVT_KEY_DOWN, self.onKeyDown)
            self.pnl.Bind(wx.EVT_LEFT_DOWN, self.onMouseDown)
            self.pnl.Bind(wx.EVT_LEFT_UP, self.onMouseUp)
            self.pnl.Bind(wx.EVT_MOTION, self.onMouseMove)
            self.Bind(wx.EVT_PAINT, self.onPaint)

            self.pout = np.zeros((0,2))
            self.pts = np.array([])
            self.pti = []

            self.srcs = srcs
            self.n = 0
            self.dn = 1
            self.ddn = 10
            self.n0 = -1
            self.N = info(srcs).N
            self.sz = info(srcs).sz
            self.Nc = self.sz[1]
            self.Nr = self.sz[0]
            self.play = False
            self.fntN = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 
                                    1, 1, thickness=2)
            self.fntT = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 
                                    1, 1, thickness=8)
            #self.r = 10
            #self.wid = 6
            self.r = 60 
            self.wid = 60

            self.SetClientSize((self.Nc,self.Nr))

            #1/0

        def onKeyDown(self, event):
            kc = event.GetKeyCode()
            #print kc
            #import pdb; pdb.set_trace()
            if kc == wx.WXK_ESCAPE :
                self.Close()
            if kc == wx.WXK_SPACE:
                self.play = not self.play
            if kc == wx.WXK_RIGHT or kc == 93: # ]
                self.n = self.n+self.ddn
            if kc == wx.WXK_LEFT or kc == 91: # [
                self.n = self.n-self.ddn
            if kc == 45: # -
                self.n = self.n - 1
            if kc == 61: # +
                self.n = self.n + 1
            if 0 <= kc - 48 <= 9 : # numbers
                k = kc - 48
                self.n = int(round((k/9.0)*self.N))

            event.Skip()

        def onMouseDown(self, event):
            pos = event.GetPosition()
            pts.append((pos[1],pos[0]))
            print pts[-1]
            #pos = np.array([[pos.x,pos.y]])
            #if self.pts.shape[0] == 0:
            #    self.pts = pos
            #    self.pti = 0
            #d = (self.pts - np.kron(np.ones((self.pts.shape[0],1)), pos))**2
            #d = d.sum(axis=1)
            #i = list((d <= self.r).nonzero()[0])
            #if not(i == []):
            #    self.pti = i[0]
            #else:
            #    self.pts = np.vstack((self.pts,pos))
            #    self.pti = self.pts.shape[0]-1
            #self.Refresh()

        def onMouseUp(self, event):
            pass
            #pos = event.GetPosition()
            #pos = (pos.x,pos.y)
            ##posc = findFeature(self.gr, pos, self.wid)
            #self.pts[self.pti,:] = np.array(pos)
            #self.pti = []
            #self.Refresh()

        def onMouseMove(self, event):
            pass
            #if not(self.pti == []):
            #    pos = event.GetPosition()
            #    self.pts[self.pti,:] = pos
            #    self.Refresh()

        def onPaint(self, event):
            dc = wx.PaintDC(self)
            dc.DrawBitmap(self.bmp, 0, 0, False)
            dc.SetBrush(wx.Brush("RED", wx.TRANSPARENT))

            for k,p in enumerate(tuple(self.pts)):
                if self.pti == k:
                    dc.SetPen(wx.Pen("GREEN", 4))
                else:
                    dc.SetPen(wx.Pen("RED", 2))
                dc.DrawEllipse(p[0]-self.r,p[1]-self.r,2.0*self.r,2.0*self.r)

        def onIdle(self, event):
            if self.play:
                self.n = self.n+self.dn

            if self.n >= self.N:
                self.n = self.N-self.dn

            if self.n < 0:
                self.n = 0

            if not (self.n == self.n0):
                img = getImg(self.srcs,self.n)
                self.displayImage(img)
                self.n0 = self.n
                #for k in range(self.pts.shape[0]):
                #    posc = findFeature(self.gr, tuple(self.pts[k,:]), self.wid)
                #    self.pts[k,:] = posc
                #    self.pout = np.vstack((self.pout,posc))
                self.pti = []
                self.Refresh()

            event.RequestMore()

        def displayImage(self, img, offset=(0,0)):
            self.SetTitle(img.pth+' : '+str(img.id))
            im = img.im
            wxi = wx.EmptyImage(im.shape[1], im.shape[0])
            if im.max() <= 1:
                im = np.array(255*im,dtype=np.uint8)
            if len(im.shape) < 3 or im.shape[2] < 3:
                im = np.dstack((im,im,im))
            im = np.array(im,dtype=np.uint8)
            wxi.SetData(im.tostring())
            self.bmp = wxi.ConvertToBitmap()

            dc = wx.ClientDC(self)
            dc.DrawBitmap(self.bmp, 0, 0, False)


    class myApp(wx.PySimpleApp):
        def OnInit(self):
            self.frame = myFrame()
            self.frame.bmp = wx.EmptyBitmap(self.frame.Nc, self.frame.Nr)
            self.frame.Show(True)
            return True

    def getBox(img, pos, wid):
        box = np.zeros((2*wid+1,2*wid+1))
        mc = np.max([-wid,-pos[0]])

    app = myApp(0)
    app.MainLoop()

    return pts


if __name__ == "__main__":
    import doctest
    doctest.testmod()


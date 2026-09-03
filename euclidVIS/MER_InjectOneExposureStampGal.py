#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#


"""
:file: python/MER_AddStamps/MER_InjectOneExposure.py

:date: 05/22/25
:author: hkong

"""

import argparse
import ElementsKernel.Logging as log
#import subprocess
import astropy.io.fits as fits
import numpy as np
import json
import xml.etree.ElementTree as ET
import galsim
import galsim
from galsim.wcs import JacobianWCS
import yaml
import os
import shutil
import fitsio

from MER_PsfMosaic.EuclidWcs import *
from MER_PsfMosaic.VisPsf import VisPsf
from MER_PsfMosaic.NirPsf import NirPsf
from MER_PsfMosaic.ExtPsf import ExtPsf
from MER_PsfMosaic.MerCatalogPsf import MerCatalogPsf
from MER_PsfMosaic.PsfExModelPsf import PsfExModelPsf
from PSFExModelModuleBinding import PsfExMasking

from MER_AddStamps.utils import multiproc, get_vis_img_list,get_vis_image_fns,get_nir_image_fns,get_nir_img_list,get_MER_MOSAIC_list
from MER_AddStamps.coverage import coverage_test
from MER_AddStamps.InjectNoise import noise_for_galaxy
from MER_AddStamps.read_image_footprint import *
from astropy.io.fits import Header

class galaxy_objects_stamp_gal():
    def __init__(self):
        self.ra = None
        self.dec = None

class sim_gal_catalog_stamp_gal(object):
    def __init__(self, wcs = None, exptime = None, zpt = None, gain = None, filter_name=None, catalog = None):
        self.wcs = wcs
        self.exptime = exptime
        self.zpt = zpt
        self.gain = gain
        self.filter_name = filter_name
        logger = log.getLogger('class: sim_gal_catalog')
        logger.info("sim_gal_catalog_stamp_gal: filter_name: %s"%(self.filter_name))
        self.catalog = catalog
        self.construct_catalog()

    def construct_catalog(self):
        radec = []
        for i in range(len(self.catalog)):
            radec.append((self.catalog['ra'][i], self.catalog['dec'][i]))
        coordxy = self.wcs.calculate_scamp_pixel_coordinates(radec)
        counts = len(self.catalog)
        objs = []
        scale_factor = self.muJy_to_calibImage_ADU( flux = 1 )
        for i in range(counts):
            obj = galaxy_objects_stamp_gal()
            #import pdb; pdb.set_trace()
            obj.stamp = self.catalog['stamp'][i]*scale_factor
            obj.bx = coordxy[i][0]
            obj.by = coordxy[i][1]
            obj.ra = self.catalog['ra'][i]
            obj.dec = self.catalog['dec'][i]
            objs.append(obj)
        self.objs = objs
        self.total = len(self.catalog)

    def muJy_to_calibImage_ADU(self, flux = None):
        """Converts object flux in a MER FINAL catalog to ADU on a VIS/NIR calib image

        Parameters
        ----------
        flux: float
        the objects flux in the MER FINAL catalog [myJy]
        exptime: float
            the exposure time of the VIS calib image (quadrant)
        zeropoint:
            the zeropoint of the VIS calib image (quadrant)

        Returns
        -------
        float
            the number of ADU on a VIS calib image
        """

        """
        NIR
        ---
        * In NIR you still can use the subroutine "muJy_to_calibImage_ADU()",
        but you need to use exptime=1.0
        * From that formula you get the integrated flux of an object in
        electrons on a NIR calibFrame
        * The RMS error in NIR need be updated as you wrote:
           WGT_new = sqrt( WGT_old**2 + (stamp value at this pixel) )
        """
        #The BUNIT keyword in the niCalibImage says "ELECTRONS" so that unit is ELECTRONS. The VIS unit is ADU.
        REFMAG_AB = 23.9
        ext_keywords = ["PANSTARRS", "MEGACAM", "HSC", "DECAM"]
        if self.filter_name == "VIS":
            return flux*self.exptime/np.power(10, 0.4*(REFMAG_AB-self.zpt))

        elif "NIR" in self.filter_name:
            return flux*1.0/np.power(10, 0.4*(REFMAG_AB-self.zpt))
        if any(k in self.filter_name for k in ext_keywords):
            return flux*1.0/np.power(10, 0.4*(REFMAG_AB-30.0))
        else:
            #some ext key words not shown
            return flux*1.0/np.power(10, 0.4*(REFMAG_AB-30.0))

def inject_gal_img_stamp_gal(img, wcs, psf, gals=None,filter_name=None, GAIN = None, Poission=False, psf_scale = None, gs_wcs = None, flippsf = False):
    full_image = np.zeros_like(img)
    full_image = galsim.Image(full_image)
    full_RMS_image = np.zeros_like(img)
    full_RMS_image = galsim.Image(full_RMS_image)

    assert(gals is not None)
    N_tot = len(gals)

    # https://github.com/GalSim-developers/GalSim/pull/450/commits/755bcfdca25afe42cccfd6a7f8660da5ecda2a65
    gsparams = galsim.GSParams(maximum_fft_size=65536)
    count = 0

    logger = log.getLogger('MER_InjectOneExposure')

    for i in range(N_tot):
        obj = gals[i]
        if filter_name == "VIS" or ("NIR" in filter_name):
            if (np.isnan(obj.bx) or np.isnan(obj.by) or obj.bx < -300 or obj.bx > 2400 or obj.by < -300 or obj.by > 2400):
                #these sources are out of boundary for this image
                continue
        local_wcs = gs_wcs.local(galsim.PositionD(obj.bx, obj.by))
        pix_scale = np.sqrt( np.abs( local_wcs._det ) )
        if filter_name == "VIS":
            rescale_factor = pix_scale/0.1
        elif "NIR" in filter_name:
            rescale_factor = pix_scale/0.05
        else:
            rescale_factor = 1
        psf_wcs = JacobianWCS( local_wcs.dudx/rescale_factor, \
                               local_wcs.dudy/rescale_factor, \
                               local_wcs.dvdx/rescale_factor, \
                               local_wcs.dvdy/rescale_factor )
        #NOTE: I assume the stamp has the same pixel resolution as the image
        #If it is in the scale of psf, then use psf_wcs instead
        gal_interp = galsim.InterpolatedImage( galsim.Image(obj.stamp) , wcs = local_wcs )

        if "NIR" in filter_name:
            img_psf = psf
            if flippsf:
                img_psf = img_psf.transpose()
        else:
            img_psf = psf.get_stamp_at_xy( (obj.bx,obj.by) ).get_data()
            if flippsf:
                img_psf = img_psf.transpose()
        psf_im = galsim.Image(img_psf/img_psf.sum(), wcs = psf_wcs)
        psf_interp= galsim.InterpolatedImage( psf_im , wcs = psf_wcs )
        gal = galsim.Convolve([gal_interp, psf_interp], gsparams=gsparams)

        bx = int(obj.bx)
        by = int(obj.by)
        d_bx = obj.bx - np.floor(obj.bx)+0.5
        d_by = obj.by - np.floor(obj.by)+0.5
        offset = galsim.PositionD(d_bx, d_by)
        stamp = gal.drawImage(method='no_pixel', offset = offset, wcs = local_wcs, nx = 174, ny = 174)
        stamp.setCenter(bx, by)
        overlap = stamp.bounds & full_image.bounds

        if overlap.area() > 0:
            count+=1
            stamp = stamp[overlap]
            if Poission is False:
                #import pdb;pdb.set_trace()
                full_image[overlap] += stamp
            if Poission:
                stamp_with_noise, RMS_Squared = noise_for_galaxy(stamp, GAIN, filter_name)
                full_image[overlap] += stamp_with_noise
                full_RMS_image[overlap] += RMS_Squared

    logger.info("inject_gal_img_stamp_gal: %d/%d sources overlap with this image"%( count, N_tot ))
    return full_image, full_RMS_image,count



def process_one_tile_stamp_gal(data_file_name, psf_file_name, new_data_file_name, catalog, filter_name, isPoisson, flippsf = False):
    logger = log.getLogger('process_one_tile')

    shutil.copyfile(data_file_name, new_data_file_name)

    with fitsio.FITS(new_data_file_name, mode='rw') as f:
        N_ccds = int((len(f) - 1) / 3)
        counts_injected_img = np.zeros(N_ccds, dtype=int)

        for ccd_idx in range(N_ccds):
            logger.info("CCD index: %d/%d", ccd_idx, N_ccds)

            if filter_name == "VIS" or "NIR" in filter_name:
                img_hdu_idx = ccd_idx*3 + 1
                rms_hdu_idx = ccd_idx*3 + 2
            else:
                img_hdu_idx = 0

            ccd_img_data = f[img_hdu_idx].read()
            ccd_img_header = f[img_hdu_idx].read_header()
            ccd_rms_data = f[rms_hdu_idx].read() if filter_name == "VIS" or "NIR" in filter_name else None

            #convert to astropy header
            astropy_hdr = Header()
            for key in ccd_img_header.keys():
                value = ccd_img_header[key]
                astropy_hdr[key] = value
            wcs = EuclidWcs(astropy_hdr)
            gs_wcs = galsim.FitsWCS(header = astropy_hdr)

            if filter_name == "VIS":
                EXPTIME = float(ccd_img_header['EXPTIME'])
                MAGZEROP = float(ccd_img_header['MAGZEROP'])
            elif "NIR" in filter_name:
                EXPTIME = 87.2448
                MAGZEROP = float(ccd_img_header['ZPAB'])
            else:
                EXPTIME = 1
                MAGZEROP = 30.0

            GAIN = float(ccd_img_header['GAIN'])

            logger.debug("exposure time: %f, zeropoint: %f, GAIN: %f", EXPTIME, MAGZEROP, GAIN)

            gals = sim_gal_catalog_stamp_gal(
                catalog=catalog,
                wcs=wcs,
                exptime=EXPTIME,
                zpt=MAGZEROP,
                gain=GAIN,
                filter_name=filter_name
            ).objs
            #import pdb;pdb.set_trace()

            if filter_name == "VIS":
                data_psf, header_psf = fits.getdata(psf_file_name, ext=ccd_idx+1, header=True)
                header_psf.set("DETECTOR", header_psf["EXTNAME"])
                logger.debug("DETECTOR: %s", header_psf['DETECTOR'])
                psf = VisPsf(data_psf, header_psf)
                psf_scale = None
            elif "NIR" in filter_name:
                #NOTE:
                #reading it directly since NIR PSF is constant; This might need to be modified for DR2
                psf = fitsio.FITS( psf_file_name)[ccd_idx+1].read()['PSF_MASK'][0][0]
                #psf = PsfExModelPsf.from_file(psf_file_name, extension=ccd_idx+1)
                psf_scale = None
            else:
                psf = ExtPsf.from_file(psf_file_name)
                psf_scale = abs(fitsio.read_header(psf_file_name, ext=1)["CD1_1"]) * 3600
                ####NOTE: adding this because Poission is not implemented for EXT!!!
                isPoisson = False

            sim_img, sim_RMS_image, injection_count = inject_gal_img_stamp_gal(
                ccd_img_data, wcs, psf, gals=gals, filter_name=filter_name,
                GAIN=GAIN, Poission=isPoisson, psf_scale=psf_scale, gs_wcs = gs_wcs, flippsf = flippsf
            )


            counts_injected_img[ccd_idx] = injection_count

            injected_img = sim_img.array + ccd_img_data
            if filter_name == "VIS" or "NIR" in filter_name:
                injected_rms_img = np.sqrt(sim_RMS_image.array + ccd_rms_data**2)
                f[img_hdu_idx].write(injected_img.astype(np.float32))
                f[rms_hdu_idx].write(injected_rms_img.astype(np.float32))
            else:
                f[img_hdu_idx].write(injected_img.astype(np.float32))

    logger.debug("Finished writing modified image to disk")
    return counts_injected_img


def _update_header(header, updates):
    for key, value in updates.items():
        if key in header:
            header[key] = value
        else:
            header.add_record({'name': key, 'value': value})
    return header

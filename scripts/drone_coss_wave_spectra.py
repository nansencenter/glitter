import os
from pathlib import Path
import glob
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, fftfreq
from scipy.signal import medfilt
from scipy.interpolate import RectBivariateSpline
from scipy.signal import welch



DIR_IN = Path('/workspaces/glitter/dev-results') / '20230105/DJI_0398' / 'projections'   # path/to/dir with drone frames
DIR_OUT = DIR_IN.parent / 'diags'
CONFIG = 'band0_sub3_res2_cubic_kernel20_dt1s'
TIME_STEP = 1       # seconds
GROUND_SPACING = 1  # meters

DIR_OUT.mkdir(exist_ok=True)

file_in = DIR_IN / 'frame_band0_sub3_res2_cubic_kernel20_dt1s_0398.nc'
frames_ds = nc.Dataset(file_in)
data = frames_ds['data'][:,::-1,:]


def rebin(a, shape):
    
    sh = shape[0], a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]    
    return a.reshape(sh).mean(-1).mean(1)

def looks2xspec(im1, im2, periodo_size):
    """
    Derive cross-spectra from a pair of frames.
    
    :param im1:
    :param im2:
    :param periodo_size:
    :returns specs:
    """
    
    imshape = np.array(im1.shape, dtype='int32')

    ###########################################################################
    # Set periodograms/looks/specs sizes and positions
    ###########################################################################

    # Define Hanning window in azimuth and range
    aziwindow = np.hanning(periodo_size + 2)[1:-1]
    ranwindow = np.hanning(periodo_size + 2)[1:-1]
    # Two dimensional window as a root of product of range and azimuth hanning windows 
    window = np.sqrt(np.outer(aziwindow, ranwindow))
    
    ###########################################################################
    # Compute periodograms and compute co/cross spectra
    ###########################################################################
    
    # Compute number of periodograms in the image 
    count = np.ceil(imshape[0]/periodo_size).astype('int32')
    # Positions of a periodogram in azimuth and range direction
    perpos = (np.floor(np.linspace(0, imshape[0]-periodo_size, num=count)+0.5).astype('int32'),
              np.floor(np.linspace(0, imshape[1]-periodo_size, num=count)+0.5).astype('int32'))
    # Define shape of the spectra
    specshape = np.array((periodo_size, periodo_size), dtype='int32')
    # Create empty spectra (filled zeros) with shape <specshape>
    specs = np.zeros(specshape, dtype='complex64')
    # For each periodogram in the image
    for appos in iter(perpos[0]):
        for rppos in iter(perpos[1]):
            per1 = fft_periodogram(im1, appos, rppos, periodo_size, window)
            per2 = fft_periodogram(im2, appos, rppos, periodo_size, window)
            specs += per1 * np.conj(per2)

    return specs

def fft_periodogram(image, az_pos, range_pos, periodo_size, window):
    # Subset periodogram from the image 1
    subset = image[az_pos:az_pos + periodo_size, range_pos:range_pos + periodo_size]
    # Remove mean
    subset = subset - np.mean(subset)
    # Calculate FFT for the preiodogram
    per = fftshift(fft2(subset * window))/periodo_size
    return per

spec = []

for i in range(data.shape[0]-1):

    data1 = data[i]
    data2 = data[i+1]
    
    # Croping
    Ny, Nx = 1, 1
    while 2*Ny<data1.shape[0]:
        Ny *= 2
    while 2*Nx<data1.shape[1]:
        Nx *= 2
    data1 = data1[int(data1.shape[0]//2 - Ny//2): int(data1.shape[0]//2 + Ny//2),
                  int(data1.shape[1]//2 - Nx//2): int(data1.shape[1]//2 + Nx//2)]
    data2 = data2[int(data2.shape[0]//2 - Ny//2): int(data2.shape[0]//2 + Ny//2),
                  int(data2.shape[1]//2 - Nx//2): int(data2.shape[1]//2 + Nx//2)]

    _spec = looks2xspec(data1, data2, min(Nx,Ny))
    spec.append(_spec)
    
spec = np.asarray(spec).sum(axis=0)/len(spec)


periodo_size = min(Nx,Ny)

xlim = (-.25,.25)
ylim = (-.25,.25)

kran = (np.arange(periodo_size)-periodo_size/2.)/periodo_size*2*np.pi/GROUND_SPACING
kazi = (np.arange(periodo_size)-periodo_size/2.)/periodo_size*2*np.pi/GROUND_SPACING


fig = plt.figure(figsize = (14,10))

plt.subplot(2, 2, 1)
plt.imshow(np.abs(spec), extent=[kran[0], kran[-1],
            kazi[-1], kazi[1]], aspect='auto',cmap='jet')
plt.xlabel('wavenumber [rad/m]')
plt.ylabel('wavenumber [rad/m]')
plt.title('Absolute value')
plt.xlim(xlim)
plt.ylim(ylim)
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(np.real(spec), extent=[kran[0], kran[-1],
            kazi[-1], kazi[1]],aspect='auto',cmap='jet')
plt.xlabel('wavenumber [rad/m]')
plt.ylabel('wavenumber [rad/m]')
plt.title('Real Part')
plt.xlim(xlim)
plt.ylim(ylim)
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(np.imag(spec), extent=[kran[0], kran[-1],
            kazi[-1], kazi[1]],aspect='auto',cmap='jet')
plt.xlabel('wavenumber [rad/m]')
plt.ylabel('wavenumber [rad/m]')
plt.title('Imaginary part')
plt.xlim(xlim)
plt.ylim(ylim)
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(np.angle(spec), extent=[kran[0], kran[-1],
           kazi[-1], kazi[1]],aspect='auto',cmap='jet')
plt.xlabel('wavenumber [rad/m]')
plt.ylabel('wavenumber [rad/m]')
plt.title('Phase')
plt.xlim(xlim)
plt.ylim(ylim)
plt.colorbar()

plt.savefig(DIR_OUT / f'diag_{CONFIG}_cross-spectra.png')


def cart2logpol(spec, kazi, kran, nphi=72, nk=60, kmin=2.*np.pi/500,
                kmax=2.*np.pi/1, heading=None):
    """
    """
    # Define log-polar grid
    phi = np.linspace(0., 360., num=nphi, endpoint=False)
    dphi = phi[1] - phi[0]
    alpha = (kmax / kmin) ** (1. / (nk - 1.))
    k = kmin * alpha ** np.arange(nk)
    dk = (np.sqrt(alpha) - 1. / np.sqrt(alpha)) * k
    # Bilinear interpolation
    intfunc = RectBivariateSpline(kazi, kran, spec, kx=1, ky=1)
    if heading is None:
        # phi clockwise, phi=0 means up azimuth
        angincart = np.deg2rad(90. - phi)
    else:
        angincart = np.deg2rad(90. - (phi - heading))
    intkazi = k[np.newaxis, :] * np.sin(angincart[:, np.newaxis])
    intkran = k[np.newaxis, :] * np.cos(angincart[:, np.newaxis])
    polspec = np.zeros((nphi, nk), dtype=spec.dtype)
    indint = np.where((intkazi >= kazi[0]) & (intkazi <= kazi[-1]) & (intkran >= kran[0]) & (intkran <= kran[-1]))
    polspec[indint] = intfunc(intkazi[indint], intkran[indint], grid=False)
    # Energy conservation
    dkazi = kazi[1] - kazi[0]
    dkran = kran[1] - kran[0]
    kcart = np.sqrt(kazi[:, np.newaxis] ** 2. + kran[np.newaxis, :] ** 2.)
    indcart = np.where((kcart >= kmin) & (kcart <= kmax))
    enecart = 4. * np.sqrt(np.nansum(np.abs(spec[indcart])) * dkazi * dkran)
    areapol = k * dk * np.deg2rad(dphi)
    enepol = 4. * np.sqrt(np.nansum(np.abs(polspec) * areapol[np.newaxis, :]))
    polspec *= (enecart / enepol) ** 2.

    return polspec, phi, dphi, k, dk

#
lmin = 50

# Spectra
polspec, phi, dphi, ks, dk = cart2logpol(np.abs(spec), kazi, kran,nphi=72*2)

# Phase
polphase, phi, dphi, ks, dk = cart2logpol(np.angle(spec), kazi, kran,nphi=72*2)

# Remove pixels with low energies or with negative phase
threshold = .0
indNaN = (polspec<threshold*polspec.max()) | (polphase<0)
polspec[indNaN] = np.nan
polphase[indNaN] = np.nan

# Polar coordinates
_pl = phi + dphi / 2.#np.concatenate(([phi[0] - dphi / 2.], phi[:-1] + dphi / 2., [phi[-1] + dphi / 2.]))
_pl = np.deg2rad(90. - _pl)
indk = np.where((ks <= 2. * np.pi / lmin))[0]
k = ks[indk] + dk[indk] / 2.
_kl = +k#np.concatenate(([0], k))
_xl = _kl[np.newaxis, :] * np.cos(_pl[:, np.newaxis])
_yl = _kl[np.newaxis, :] * np.sin(_pl[:, np.newaxis])


fig = plt.figure(figsize=(15,10))
plt.subplots_adjust(left=0.15, right=0.975, bottom=0.1, top=0.95, wspace=0.145)
plt.pcolormesh(_xl, _yl, polspec[:, indk], cmap=plt.get_cmap('jet'))#, vmin=vmin, vmax=vmax)

# Iso-angles
cirx = np.cos(np.linspace(0, 2. * np.pi, num=100))
ciry = np.sin(np.linspace(0, 2. * np.pi, num=100))
for wl in [500, 200, 100, 60]:
    plt.plot(2. * np.pi / wl * cirx, 2. * np.pi / wl * ciry,color='silver',linewidth=0.6 )
    plt.text(0, 2. * np.pi / wl,'{:d}m'.format(wl),color='k',horizontalalignment='center', fontsize=20)


plt.xlabel('Range wavenumber (rad/m)', fontsize=20)
plt.ylabel('Azimuth wavenumber (rad/m)', fontsize=20)
plt.gca().set_aspect('equal')
cbar = plt.colorbar()

# change colorbar fontsize
tick_font_size = 20
cbar.ax.tick_params(labelsize=tick_font_size)

# changing the fontsize of yticks
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.show()

fig.savefig(DIR_OUT / f'diag_{CONFIG}__polar-cross-spectra.png')

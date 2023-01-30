import os
from pathlib import Path
from datetime import datetime
import numpy as np
import netCDF4 as nc
from matplotlib.image import imread
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# Config processing sparameters
DIR_IN = Path('/workspaces/glitter/dev-data/drone') / '20230105/DJI_0398'   # path/to/dir with drone frames
DIR_OUT = Path('/workspaces/glitter/dev-results') / '20230105/DJI_0398' / 'projections'    # path/to/dir to save projected frames
NAME_OUT = 'frame'                  #
BAND = 0                            # ID of RGB band
SUBSAMPLING = 3                     # Every N pixel in x and y
RES_REG = 2                         # Final pixel size in meters
KERNEL_SIZE = 20                    # Kernel size for the gaussian filter
METHOD = 'cubic'                    # Grid interpolation method
TIME_STEP = 1                       # Time diff between two frames in seconds
H_DRONE = 300                       # Drone camera height above ground


def projection(x, y, h, alpha):
    """
    Reproject data from the original (drone) x, y grid to ground grid
    using camera height over ground and angle resolution of the camera.
    
    :param x: 2d array, with range coordinates in meters
    :param y: 2d array, with azimuth coordinates in meters 
    :param h: float, height of the camera (drone) over ground in meters
    :param alpha: float, angular resolution of the camera in rad.
    :returns x_proj: 2d array, projected x coordinates on ground range in meters
    :returns y_proj: 2d array, projected y coordinates on ground range in meters
    """
    x_proj = np.tan(x * alpha) * h
    y_proj = np.tan(y * alpha) * h
    return x_proj, y_proj


def process_one_frame(
    file_img,
    h_drone,
    band=1,
    subsampling=1,
    alpha=0.0005251,
    res_reg=1,
    kernel_size=40,
    interp_method='linear'):
    """
    Read, normalize, and project a frame. 
    
    NOTE: Projection of data from camera grid to regular ground range grid:
        1. Create camera grid (for original image) using angular resolution and size of the frame
        2. Create ground range (irregular) grid and project data from the camera grid (see 1)
        3. Project from the ground range grid to the regular grid
        
    NOTE: Final grid is meters with x = y = 0 in the center of the frame
        
    :param file_img: str, path to the frame
    :param h_drone: height of the camera (drone) in meters
    :param band: int, 0,1, or 2 for RGB bands. NOTE: check band ids
    :param subsampling: subsampling factor (use every n pix in both x and y)
    :param alpha: angular resolution of the camera in rad. 
    :param res_reg:
    :param kernel_size: ???
    :param interp_method: method for griddata function
    :returns x_reg: ground range x coordinates in meters
    :returns y_reg: ground range y coordinates in meters
    :returns data_corrected: reprojected data from the band
    """
    # Read image file
    rgb_img = imread(file_img)
    # image values
    data = rgb_img[:,:, band]
    # subsampling
    data = convolve2d(data,np.ones((subsampling,subsampling))/subsampling**2)
    data = data[::subsampling, ::subsampling]
    # Generate camera grid
    y_img = np.arange(-data.shape[0]/2, data.shape[0]/2)
    x_img = np.arange(-data.shape[1]/2, data.shape[1]/2)
    # Project from camera grid to ground coordinates
    x_grnd, y_grnd = projection(x_img, y_img, h=h_drone, alpha=subsampling*alpha)
    # Create a ground range x and y mesh
    x2d_grnd, y2d_grnd = np.meshgrid(x_grnd, y_grnd)
    # Interpolate to regular grid
    # Generate 1d x and y regular-grid coordinates
    x_reg = np.arange(x_grnd[0], x_grnd[-1], res_reg)
    y_reg = np.arange(y_grnd[0], y_grnd[-1], res_reg)
    # Create regular x and y mesh
    x2d_reg, y2d_reg = np.meshgrid(x_reg, y_reg)
    # Linear interpolation
    # Inter polate data from original camera grid to ground range grid using linear interpolation
    data_interp = griddata((x2d_grnd.flatten(), y2d_grnd.flatten()), data.flatten(),
                           (x2d_reg.flatten(),y2d_reg.flatten()), method=interp_method).reshape(x2d_reg.shape)
    # Correct for sun glint using gaussian filter
    if kernel_size is not None:
        glint = gaussian_filter(data_interp, sigma=kernel_size)
        data_corrected = data_interp/glint
    else: 
        data_corrected = +data_interp
    
    return x_reg, y_reg, data_corrected


os.makedirs(DIR_OUT, exist_ok=True)
# input files
files_in = list(DIR_IN.glob('*.png'))
num = files_in[0].parts[-2].split("_")[-1]
out_fname = f'{NAME_OUT}_band{BAND}_sub{SUBSAMPLING}_res{RES_REG}_{METHOD}_kernel{KERNEL_SIZE}_dt{TIME_STEP}s_{num}.nc'
dst = DIR_OUT / out_fname
if dst.exists():
    dst.unlink()

# Stack processed frames
data_stack = []
# Track processing time
start = datetime.now()
for i in range(len(files_in)):
    print(f'{i + 1} / {len(files_in)}', end=' | ')
    # Preprocess drone frame
    x, y, data = process_one_frame(
        files_in[i],
        h_drone=H_DRONE, 
        band=BAND,
        subsampling=SUBSAMPLING, 
        res_reg=RES_REG, 
        kernel_size=KERNEL_SIZE,
        interp_method=METHOD
    )
    # Add procesed data to the stack
    data_stack.append(data)
    print(f'{datetime.now() - start}', end='\r')

print(out_fname)
# Save to netcdf
ds = nc.Dataset(dst, mode='w')
ds.createDimension('time', len(files_in))
ds.createDimension('x', x.size)
ds.createDimension('y', y.size)
var = ds.createVariable('time', 'f4', dimensions=('time'))
var[:] =  np.arange(len(files_in)) * TIME_STEP
var = ds.createVariable('data', 'f8', dimensions=('time','y','x'))
var[:] = data_stack
ds.close()
          
print('DONE')

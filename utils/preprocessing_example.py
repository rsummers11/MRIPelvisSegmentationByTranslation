# %%
# This code used the Gold Atlas - Male Pelvis dataset as an example to preprocess CT and MR
# Package version:
# python == 3.9.16
# scikit-image == 0.20.0
# simpleitk == 2.2.1

# %%
import numpy as np
import os
from skimage.io import imsave
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.color import gray2rgb
import SimpleITK as sitk

# %%
from utils import windowing

# %%
# specify path to CT and MR image
ct_prefix = 'CT.nii.gz'; mr_prefix = 'MR_t2.nii'
cur_ct = sitk.ReadImage('./CT.nii.gz')
## CT dataset, read the 3d ct scan
# change orientation
ct_sitk_reoriented = sitk.DICOMOrient(cur_ct, 'LPS')
# # convert to numpy array
ct_np = sitk.GetArrayFromImage(ct_sitk_reoriented)
ct_windowing = windowing(ct_np, [-160, 240])
cur_ct_np_new_norm = ct_windowing/255.    

## MRI dataset, load mri data
cur_mr = sitk.ReadImage('MR_t2.nii')
mr_sitk_reoriented = sitk.DICOMOrient(cur_mr, 'LPS')
mr_np = sitk.GetArrayFromImage(mr_sitk_reoriented)
mri_norm = rescale_intensity(mr_np, in_range=tuple(np.percentile(mr_np, (1, 99))))

# check total number of slices: total_index
if cur_ct_np_new_norm.shape[0] == mri_norm.shape[0]:
    total_index = cur_ct_np_new_norm.shape[0]
else:
    print('CT slices not equal to MR slices!')
    total_index = min(cur_ct_np_new_norm.shape[0], mri_norm.shape[0])

save_path = './png'
os.makedirs(save_path, exist_ok = True)
for each_frame_index in range(total_index):
    each_frame_ct = cur_ct_np_new_norm[each_frame_index,:,:]
    each_frame_ct_resized = gray2rgb(resize(each_frame_ct, (256, 256), anti_aliasing=True))
    each_frame_mri = mri_norm[each_frame_index,:,:]
    each_frame_mri_resized = gray2rgb(resize(each_frame_mri, (256, 256), anti_aliasing=True))
    imsave(os.path.join(save_path, 'ct_slice_'+str(each_frame_index)+'.png'), each_frame_ct_resized) 
    imsave(os.path.join(save_path, 'mr_slice_'+str(each_frame_index)+'.png'), each_frame_mri_resized)
    
    #break



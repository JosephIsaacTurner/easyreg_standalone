import nilearn.image as nli
import numpy as np
import nibabel as nib
import os
from .mri_easyreg import register_images

fs_home = os.path.dirname(os.path.realpath(__file__))

def img_to_2mm(roi_reg_path, output_path):
    """
    Resample the roi_reg to 2mm resolution (91, 109, 91).
    """
    if not os.path.exists(roi_reg_path):
        raise ValueError("roi_reg_path does not exist")
    roi_reg = nib.load(roi_reg_path)
    affine = np.array([[  -2.,    0.,    0.,   90.],
       [   0.,    2.,    0., -126.],
       [   0.,    0.,    2.,  -72.],
       [   0.,    0.,    0.,    1.]])
    target = nib.Nifti1Image(np.zeros((91, 109, 91)), affine)
    two_mm_img = nli.resample_to_img(roi_reg, target, interpolation="nearest")
    two_mm_img_data = two_mm_img.get_fdata()
    two_mm_img_data[two_mm_img_data < 0.9] = 0
    two_mm_img_data[two_mm_img_data >= 0.9] = 1
    two_mm_img = nib.Nifti1Image(two_mm_img_data, affine)
    two_mm_img.to_filename(output_path)

def register_images_mni_152(flo, 
                            flo_seg, 
                            ref=os.path.join(fs_home, 'mni152_mricron.nii.gz'), 
                            ref_seg=os.path.join(fs_home, 'mni152_mricron_seg.nii.gz'), 
                            ref_reg=None, 
                            flo_reg=None, 
                            fwd_field=None, 
                            bak_field=None, 
                            affine_only=False, 
                            threads=1,
                            autocrop=False):
    register_images(ref=ref, 
                    ref_seg=ref_seg, 
                    flo=flo, 
                    flo_seg=flo_seg, 
                    ref_reg=ref_reg, 
                    flo_reg=flo_reg, 
                    fwd_field=fwd_field, 
                    bak_field=bak_field, 
                    affine_only=affine_only, 
                    autocrop=autocrop, 
                    threads=threads)
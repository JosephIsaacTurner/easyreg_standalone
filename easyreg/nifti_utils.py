import nilearn.image as nli
import numpy as np
import nibabel as nib
import os
from .mri_easyreg import register_images
from .mri_easywarp import warp_image

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
    two_mm_img = nib.Nifti1Image(two_mm_img_data, affine)
    two_mm_img.to_filename(output_path)

def register_image_mni152(flo, 
                            flo_seg, 
                            ref=os.path.join(fs_home, 'mni152_mricron.nii.gz'), 
                            ref_seg=os.path.join(fs_home, 'mni152_mricron_seg.nii.gz'), 
                            ref_reg=None, 
                            flo_reg=None, 
                            fwd_field=None, 
                            bak_field=None, 
                            affine_only=False, 
                            threads=1,
                            autocrop=True):
    print("Registering image to MNI152 space", flush=True)
    print("Using {} threads".format(threads), flush=True)
    print("With autocrop: {}".format(autocrop), flush=True)
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
    
def segment_image_mni152(flo,
                        flo_seg, # Required
                        ref=os.path.join(fs_home, 'mni152_mricron.nii.gz'),
                        ref_seg=os.path.join(fs_home, 'mni152_mricron_seg.nii.gz'),
                        ref_reg=None,
                        flo_reg=None, # required
                        fwd_field=None, # requierd
                        bak_field=None, 
                        affine_only=False,
                        threads=1,
                        autocrop=True,
                        post=False):
    if flo_seg is None:
        raise ValueError("flo_seg is required")
    if flo_reg is None:
        raise ValueError("flo_reg is required")
    if fwd_field is None:
        raise ValueError("fwd_field is required")
    print("Segmenting image to MNI152 space", flush=True)
    print("Using {} threads".format(threads), flush=True)
    print("With autocrop: {}".format(autocrop), flush=True)
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
                    threads=threads,
                    post=post)
    warp_image(
        input_image=flo_seg,
        output_image=flo_seg.replace('.nii', '_mni152.nii'),
        field=fwd_field,
        nearest=True,
        threads=threads
    )
    if post:
        warp_image(
            input_image=flo_seg.replace('.nii', '_posteriors.nii'),
            output_image=flo_seg.replace('.nii', '_mni152_posteriors.nii'),
            field=fwd_field,
            threads=threads,
        )
    
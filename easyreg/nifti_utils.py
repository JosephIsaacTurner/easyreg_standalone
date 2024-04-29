import nilearn.image as nli
import numpy as np
import nibabel as nib

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

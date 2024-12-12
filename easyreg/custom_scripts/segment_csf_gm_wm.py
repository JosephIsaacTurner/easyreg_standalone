import numpy as np
from nilearn.maskers import NiftiMasker
from easyreg import segment_image_mni152
from nilearn.image import smooth_img
import argparse

def segment_t1(raw_img_path):
    """
    Function to coregister a T1 image to MNI152 space and segment it into GM, WM, and CSF images.
    First, the image is registered to MNI152 space using a 1mm MNI152 template, with easyreg.
    Then, the image is segmented into GM, WM, and CSF using the 32 labels, with synthseg.
    These images are then smoothed with a 2mm FWHM Gaussian kernel.
    """

    # Define the paths for the output images.
    raw_img_seg = raw_img_path.replace('.nii', '_seg.nii') # Native space segmentation into 32 synthseg labels.
    raw_img_reg = raw_img_path.replace('.nii', '_mni152.nii') # Registered image to MNI152 space.
    raw_img_fwd_field = raw_img_path.replace('.nii', '_fwd_field.nii') # Forward deformation field.
    img_posterios_mni = raw_img_path.replace('.nii', '_seg_mni152_posteriors.nii') # Segmentation into 32 labels in MNI152 space.
    gm_img_path = raw_img_path.replace('.nii', '_smwp1.nii') # Smoothened GM segmentation in MNI152 space.
    wm_img_path = raw_img_path.replace('.nii', '_smwp2.nii') # Smoothened WM segmentation in MNI152 space.
    csf_img_path = raw_img_path.replace('.nii', '_smwp3.nii') # Smoothened CSF segmentation in MNI152 space.

    # Register the image to MNI152 space using a 1mm MNI152 template.
    segment_image_mni152(
        flo=raw_img_path,
        flo_seg=raw_img_seg,
        flo_reg=raw_img_reg,
        fwd_field=raw_img_fwd_field,
        threads=11,
        post=True
    )

    # Segment the image into GM, WM, and CSF.
    csf_image = extract_csf_labels(img_posterios_mni)
    csf_image = smooth_img(csf_image, fwhm=(2,2,2))
    csf_image.to_filename(csf_img_path)

    gm_image = extract_gm_labels(img_posterios_mni)
    gm_image = smooth_img(gm_image, fwhm=(2,2,2))
    gm_image.to_filename(gm_img_path)

    wm_image = extract_wm_labels(img_posterios_mni)
    wm_image = smooth_img(wm_image, fwhm=(2,2,2))
    wm_image.to_filename(wm_img_path)

    compute_deterministic_atlas(raw_img_path, gm_img_path, wm_img_path, csf_img_path)


def extract_csf_labels(path):
    """
    From the 32 labels, we select the ones that correspond to CSF.
    Ventricles are considered CSF.
    """
    masker = NiftiMasker(mask_img='MNI152_T1_1mm_brain_mask_dil.nii.gz')
    data = masker.fit_transform(path)
    data_csf = data[[3,4,11,12,16,21,22],: ]
    data_csf = np.nan_to_num(data_csf)
    data_csf = np.max(data_csf, axis=0)
    data_csf = masker.inverse_transform(data_csf)
    return data_csf


def extract_gm_labels(path):
    """
    From the 32 labels, we select the ones that correspond to GM.
    """
    masker = NiftiMasker(mask_img='MNI152_T1_1mm_brain_mask_dil.nii.gz')
    data = masker.fit_transform(path)
    data_csf = data[[2,6,7,8,9,10,14,15,17,20,24,25,26,27,28,29,30,31],: ]
    data_csf = np.max(data_csf, axis=0)
    data_csf = masker.inverse_transform(data_csf)
    return data_csf


def extract_wm_labels(path):
    """
    From the 32 labels, we select the ones that correspond to WM. 
    Brainstem and tectum are considered WM.
    """
    masker = NiftiMasker(mask_img='MNI152_T1_1mm_brain_mask_dil.nii.gz')
    data = masker.fit_transform(path)
    data_csf = data[[1,5,13,18,19,23,32],: ]
    data_csf = np.max(data_csf, axis=0)
    data_csf = masker.inverse_transform(data_csf)
    return data_csf


def compute_deterministic_atlas(raw_img_path, gm_img_path, wm_img_path, csf_img_path):
    """
    Function to compute the deterministic atlas from the GM, WM, and CSF images.
    By design, there will be no overlap between the three labels.
    """
    masker = NiftiMasker(mask_img='MNI152_T1_1mm_brain_mask_dil.nii.gz')
    data = masker.fit_transform([gm_img_path, wm_img_path, csf_img_path])
    data = np.argmax(data, axis=0)
    data += 1 # That way the labels are 1, 2, 3
    data = masker.inverse_transform(data)
    data.to_filename(raw_img_path.replace('.nii', '_deterministic_atlas.nii'))


def main():
    """
    Main function to parse the arguments and call the segment_t1 function.
    """
    parser = argparse.ArgumentParser(description='Segment a T1 image into GM, WM, and CSF images.')
    parser.add_argument('img_path', type=str, help='Path to the raw T1 image.')
    args = parser.parse_args()
    segment_t1(args.img_path)

if __name__ == '__main__':
    main()
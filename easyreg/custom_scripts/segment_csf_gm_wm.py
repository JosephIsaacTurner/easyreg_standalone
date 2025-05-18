import os
import shutil
import argparse
from pathlib import Path
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn.image import smooth_img
from easyreg import segment_image_mni152

EASYREG_DIR = Path(__file__).resolve().parent.parent


def orchestrate_csf_mapping(
    raw_img_path: str,
    ref_template_path: str = None,
    ref_template_seg_path: str = None,
    output_prefix: str = None,
    threads: int = 11,
) -> dict:
    """
    Register a T1 image to template space, segment into GM/WM/CSF for CSF mapping, smooth the maps,
    and compute a deterministic atlas for CSF mapping.

    Parameters
    ----------
    raw_img_path : str
        Path to the raw T1 image.
    ref_template_path : str, optional
        Reference template image path (default: MNI152 in EASYREG_DIR).
    ref_template_seg_path : str, optional
        Reference template segmentation path.
    output_prefix : str, optional
        Directory or file prefix for outputs.
    threads : int, default 11
        Number of threads for registration.

    Returns
    -------
    dict
        Paths to segmentation and CSF mapping outputs.
    """
    # Resolve template paths
    if not ref_template_path:
        ref_template_path = str(EASYREG_DIR / "mni152_mricron.nii.gz")
        if not ref_template_seg_path:
            ref_template_seg_path = str(EASYREG_DIR / "mni152_mricron_seg.nii.gz")
        print(f"Using default template: {ref_template_path}")
        if ref_template_seg_path == str(EASYREG_DIR / "mni152_mricron_seg.nii.gz"):
            print(f"Using default template segmentation: {ref_template_seg_path}")
    elif not ref_template_seg_path:
        print(
            f"Custom template {ref_template_path} provided without a segmentation path."
        )

    # Determine output base
    raw = Path(raw_img_path)
    name_stem = raw.name
    if name_stem.endswith(".nii.gz"):
        name_stem = name_stem[: -len(".nii.gz")]
    elif name_stem.endswith(".nii"):
        name_stem = name_stem[: -len(".nii")]

    if output_prefix:
        outp = Path(output_prefix)
        if str(output_prefix).endswith(os.sep) or (outp.exists() and outp.is_dir()):
            out_dir, base_filename_prefix = outp, name_stem
        else:
            out_dir, base_filename_prefix = outp.parent, outp.name
    else:
        out_dir, base_filename_prefix = raw.parent, name_stem

    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = str(out_dir / base_filename_prefix)

    # Define native segmentation path first
    native_seg_path = f"{base_path}_seg.nii"

    outputs = {
        "native_seg": native_seg_path,
        "registered": f"{base_path}_registered_to_ref.nii",
        "field": f"{base_path}_fwd_field.nii",
        # Correctly derive posteriors path from native_seg_path
        "post": native_seg_path.replace(".nii", "_mni152_posteriors.nii"),
        "gm": f"{base_path}_smwp1_ref.nii",
        "wm": f"{base_path}_smwp2_ref.nii",
        "csf": f"{base_path}_smwp3_ref.nii",
        "atlas": f"{base_path}_deterministic_atlas_ref.nii",
    }

    print(f"--- Orchestrating CSF Mapping ---")
    print(f"Input T1 Image: {raw_img_path}")
    print(f"Output base for files: {base_path}")
    print(f"Expected warped posteriors at: {outputs['post']}")

    segment_image_mni152(
        flo=str(raw),
        ref=ref_template_path,
        ref_seg=ref_template_seg_path,
        flo_seg=outputs["native_seg"],
        flo_reg=outputs["registered"],
        fwd_field=outputs["field"],
        threads=threads,
        post=True,
        autocrop=True,
    )

    post_path = outputs["post"]
    if not Path(post_path).exists():
        raise FileNotFoundError(f"Missing expected warped posteriors file: {post_path}")

    # Extract, smooth, and save tissue maps
    for key, func in [
        ("csf", extract_csf_labels),
        ("gm", extract_gm_labels),
        ("wm", extract_wm_labels),
    ]:
        print(f"Processing {key.upper()} for CSF mapping...")
        img = func(post_path)
        smooth_img(img, fwhm=(2, 2, 2)).to_filename(outputs[key])
        print(f"{key.upper()} map saved to: {outputs[key]}")

    print("Computing deterministic atlas for CSF mapping...")
    dummy_input = f"{base_path}_dummy_for_atlas.nii"
    compute_deterministic_atlas(
        dummy_input, outputs["gm"], outputs["wm"], outputs["csf"]
    )
    generated_atlas = dummy_input.replace(".nii", "_deterministic_atlas.nii")
    if Path(generated_atlas).exists():
        shutil.move(generated_atlas, outputs["atlas"])
        print(f"Deterministic atlas saved to: {outputs['atlas']}")
    else:
        print(f"Warning: {generated_atlas} not found; skipping move")

    print("--- CSF mapping complete ---")
    return outputs


def segment_t1(raw_img_path: str) -> dict:
    """Deprecated. Segments a T1 image in MNI152 space. Calls orchestrate_csf_mapping with defaults."""
    print("segment_t1: DEPRECATED. Using MNI152 defaults.")
    mni_t = str(EASYREG_DIR / "mni152_mricron.nii.gz")
    mni_s = str(EASYREG_DIR / "mni152_mricron_seg.nii.gz")
    return orchestrate_csf_mapping(raw_img_path, mni_t, mni_s)


def _extract_tissue_labels(path: str, label_indices: list):
    """Helper to load SynthSeg posteriors and return combined probability map for CSF mapping."""
    mask = str(EASYREG_DIR / "MNI152_T1_1mm_brain_mask_dil.nii.gz")
    if not Path(mask).exists():
        raise FileNotFoundError(f"Mask not found: {mask}")
    masker = NiftiMasker(mask_img=mask, standardize=False, memory_level=1, verbose=0)
    data = masker.fit_transform(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    valid = [i for i in label_indices if i < data.shape[0]]
    if not valid:
        return masker.inverse_transform(np.zeros(data.shape[1]))
    if len(valid) < len(label_indices):
        print(f"Warning: some indices out of bounds. Using {valid}")
    selected = data[valid, :]
    prob_map = np.nan_to_num(selected).max(axis=0)
    return masker.inverse_transform(prob_map)


def extract_csf_labels(path: str):
    """Load SynthSeg posteriors and return a CSF probability map."""
    return _extract_tissue_labels(path, [3, 4, 11, 12, 16, 21, 22])


def extract_gm_labels(path: str):
    """Load SynthSeg posteriors and return a GM probability map."""
    return _extract_tissue_labels(
        path, [2, 6, 7, 8, 9, 10, 14, 15, 17, 20, 24, 25, 26, 27, 28, 29, 30, 31]
    )


def extract_wm_labels(path: str):
    """Load SynthSeg posteriors and return a WM probability map."""
    return _extract_tissue_labels(path, [1, 5, 13, 18, 19, 23, 32])


def compute_deterministic_atlas(
    dummy_raw_img_path: str, gm_img_path: str, wm_img_path: str, csf_img_path: str
):
    """
    Label each voxel by the tissue class with highest probability (1=GM, 2=WM, 3=CSF) for CSF mapping.
    `dummy_raw_img_path` forms the output filename.
    """
    mask = str(EASYREG_DIR / "MNI152_T1_1mm_brain_mask_dil.nii.gz")
    masker = NiftiMasker(mask_img=mask, standardize=False, memory_level=1, verbose=0)
    gm = np.squeeze(masker.fit_transform(gm_img_path))
    wm = np.squeeze(masker.fit_transform(wm_img_path))
    csf =np.squeeze(masker.fit_transform(csf_img_path))
    gm = gm.reshape(-1, 1) if gm.ndim == 1 else gm
    wm = wm.reshape(-1, 1) if wm.ndim == 1 else wm
    csf = csf.reshape(-1, 1) if csf.ndim == 1 else csf
    atlas = np.argmax(np.concatenate([gm, wm, csf], axis=1), axis=1) + 1
    atlas_2d = atlas[np.newaxis, :]
    out = dummy_raw_img_path.replace(".nii", "_deterministic_atlas.nii")
    masker.inverse_transform(atlas_2d).to_filename(out)
    print(f"Atlas saved to: {out}")


def main():
    """CLI for segmentation and CSF mapping."""
    parser = argparse.ArgumentParser(
        description="Segment T1, generate tissue maps, and CSF mapping."
    )
    parser.add_argument("img_path", help="Path to input T1 image (NIFTI).")
    parser.add_argument(
        "--ref_template", help="Custom reference template path.", default=None
    )
    parser.add_argument(
        "--ref_template_seg", help="Custom template segmentation path.", default=None
    )
    parser.add_argument(
        "--output_prefix", help="Output prefix or directory.", default=None
    )
    parser.add_argument(
        "--threads", type=int, help="Threads for processing.", default=11
    )
    args = parser.parse_args()
    print(f"EASYREG_DIR: {EASYREG_DIR}")
    orchestrate_csf_mapping(
        raw_img_path=args.img_path,
        ref_template_path=args.ref_template,
        ref_template_seg_path=args.ref_template_seg,
        output_prefix=args.output_prefix,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()

## EasyReg Standalone
**Joseph Turner**

**Standalone version of EasyReg, a tool for registering 3D images. It is essentially identical to FreeSurfer's EasyReg, but it is not dependent on FreeSurfer.**

### Details

Useful for those who do not have FreeSurfer installed, or who want to use EasyReg in a different context. Also easier to modify and extend. Added features include the ability to import it as a module, and an option for registration into the MNI152 template space without needing to specify the template files.

### Installation

**Important Advisories**
- If you are using a MacBook with Apple Silicon (M1/M2/M3), ensure that your TensorFlow installation is compatible with ARM architecture.
= Requires TensorFlow version 2; TensorFlow version 3 is not supported.

To install, simply clone the repository and run the setup script:

```bash
git clone https://github.com/josephisaacturner/easyreg_standalone.git
cd easyreg_standalone
pip install -e .
```

Or use pip:

```bash
pip install git+https://github.com/JosephIsaacTurner/easyreg_standalone.git
```

### Usage

**Command Line:**

You can use EasyReg from the command line with the following syntax:

```bash
easyreg-mri --ref <reference_image> --flo <floating_image> --ref_seg <reference_segmentation> --flo_seg <floating_segmentation> [optional arguments]
```

Optional arguments include:
- `--ref_reg <registered_reference>`: File for registered reference image (nifti format)
- `--flo_reg <registered_floating>`: File for registered floating image (nifti format)
- `--fwd_field <forward_field>`: File for forward deformation field (saved as a nifti file, useful if you want to apply the same transformation to other images)
- `--bak_field <backward_field>`: File for inverse deformation field (saved as a nifti file)
- `--affine_only <affine_only>`: Skips nonlinear registration and only performs affine registration if specified
- `--threads <number>`: Number of threads to use for registration

**Module:**

You can also use EasyReg as a module in your Python code:

```python
from easyreg import register_images
register_images(
    ref='path/to/reference.img',
    flo='path/to/floating.img',
    ref_seg='path/to/reference_seg.img',
    flo_seg='path/to/floating_seg.img',
    ref_reg='path/to/registered_reference.img',
    flo_reg='path/to/registered_floating.img',
    fwd_field='path/to/forward_field.img',
    bak_field='path/to/backward_field.img',
    affine_only=True,
    threads=4
)
```

### Support

For any issues or questions, please open an issue on the GitHub repository.
import os
import argparse
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F

def main():

    parser = argparse.ArgumentParser(description="EasyReg (warping code): deep learning registration simple and easy", epilog='\n')

    # input/outputs
    parser.add_argument("--i", help="Input image")
    parser.add_argument("--o", help="Output (deformed) image")
    parser.add_argument("--field", help="Deformation field")
    parser.add_argument("--nearest", action="store_true", help="(optional) Use nearest neighbor (rather than linear) interpolation")
    parser.add_argument("--threads", type=int, default=1, help="(optional) Number of cores to be used. Default is 1. You can use -1 to use all available cores")

    # parse commandline
    args = parser.parse_args()

    i = args.get('i', None)
    o = args.get('o', None)
    field = args.get('field', None)
    nearest = args.get('nearest', False)
    threads = args.get('threads', 1)

    warp_image(i, o, field, nearest=nearest, threads=threads)

# def warp_image(input_image, output_image, field, nearest=False, threads=1):
#     # Functionality extracted from the main function, to be used programmatically
#     if input_image is None or output_image is None or field is None:
#         raise ValueError("Input image, output image, and field must all be provided")
    
#     # Thread handling
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     if threads == 1:
#         print('using 1 thread')
#     elif threads < 0:
#         threads = os.cpu_count()
#         print(f'using all available threads ({threads})')
#     else:
#         print(f'using {threads} threads')
#     torch.set_num_threads(threads)

#     print('Reading deformation field')
#     field_buffer, field_aff, field_h = load_volume(field, im_only=False, squeeze=True, dtype=None)
#     if len(field_buffer.shape) !=4:
#         print('field must be 4D array')
#     if field_buffer.shape[3] != 3:
#         print('field must have 3 frames')

#     print('Reading input image')
#     input_buffer, input_aff, input_h = load_volume(input_image, im_only=False, squeeze=True, dtype=None)

#     print('Deforming (interpolating)')
#     affine = torch.tensor(np.linalg.inv(input_aff), device='cpu')
#     field_buffer = torch.tensor(field_buffer, device='cpu')
#     II = affine[0, 0] * field_buffer[:,:,:,0]  + affine[0, 1] * field_buffer[:,:,:,1]  + affine[0, 2] * field_buffer[:,:,:,2]  + affine[0, 3]
#     JJ = affine[1, 0] * field_buffer[:,:,:,0]  + affine[1, 1] * field_buffer[:,:,:,1]  + affine[1, 2] * field_buffer[:,:,:,2]  + affine[1, 3]
#     KK = affine[2, 0] * field_buffer[:,:,:,0]  + affine[2, 1] * field_buffer[:,:,:,1]  + affine[2, 2] * field_buffer[:,:,:,2]  + affine[2, 3]

#     if nearest:
#         Y = fast_3D_interp_torch(torch.tensor(input_buffer, device='cpu', requires_grad=False), II, JJ, KK, 'nearest')
#     else:
#         Y = fast_3D_interp_torch(torch.tensor(input_buffer, device='cpu', requires_grad=False), II, JJ, KK, 'linear')

#     print('Saving to disk')
#     save_volume(Y.numpy(), field_aff, field_h, output_image)

#     print('All done!')
    
# def warp_image(input_image, output_image, field, nearest=False, threads=1):
#     """
#     Warps a 3D or 4D input image using a 4D deformation field and saves the output image.

#     Parameters:
#     - input_image: Path to the input image file.
#     - output_image: Path where the warped image will be saved.
#     - field: Path to the 4D deformation field file.
#     - nearest: Boolean indicating whether to use nearest-neighbor interpolation. Defaults to False (linear).
#     - threads: Number of CPU threads to use. Defaults to 1.
#     """
#     if input_image is None or output_image is None or field is None:
#         raise ValueError("Input image, output image, and field must all be provided")
    
#     # Thread handling
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
#     if threads == 1:
#         print('Using 1 thread')
#     elif threads < 0:
#         threads = os.cpu_count()
#         print(f'Using all available threads ({threads})')
#     else:
#         print(f'Using {threads} threads')
#     torch.set_num_threads(threads)

#     print('Reading deformation field...')
#     field_buffer, field_aff, field_h = load_volume(field, im_only=False, squeeze=True, dtype=None)
    
#     if field_buffer.ndim != 4:
#         raise ValueError('Deformation field must be a 4D array')
#     if field_buffer.shape[3] != 3:
#         raise ValueError('Deformation field must have 3 components in the last dimension')
    
#     print('Reading input image...')
#     input_buffer, input_aff, input_h = load_volume(input_image, im_only=False, squeeze=True, dtype=None)
    
#     # Determine if the input image is 3D or 4D
#     if input_buffer.ndim == 3:
#         input_tensor = torch.tensor(input_buffer, device='cpu', dtype=torch.float32).unsqueeze(0)  # Add channel dimension
#     elif input_buffer.ndim == 4:
#         input_tensor = torch.tensor(input_buffer, device='cpu', dtype=torch.float32)
#     else:
#         raise ValueError('Input image must be either 3D or 4D')

#     print('Preparing deformation field...')
#     affine = torch.tensor(np.linalg.inv(input_aff), device='cpu', dtype=torch.float32)
#     field_tensor = torch.tensor(field_buffer, device='cpu', dtype=torch.float32)
    
#     # Apply affine transformation to the deformation field
#     II = affine[0, 0] * field_tensor[:, :, :, 0] + affine[0, 1] * field_tensor[:, :, :, 1] + affine[0, 2] * field_tensor[:, :, :, 2] + affine[0, 3]
#     JJ = affine[1, 0] * field_tensor[:, :, :, 0] + affine[1, 1] * field_tensor[:, :, :, 1] + affine[1, 2] * field_tensor[:, :, :, 2] + affine[1, 3]
#     KK = affine[2, 0] * field_tensor[:, :, :, 0] + affine[2, 1] * field_tensor[:, :, :, 1] + affine[2, 2] * field_tensor[:, :, :, 2] + affine[2, 3]
    
#     print('Performing interpolation...')
#     if nearest:
#         warped = fast_3D_interp_torch(input_tensor, II, JJ, KK, mode='nearest')
#     else:
#         warped = fast_3D_interp_torch(input_tensor, II, JJ, KK, mode='linear')
    
#     # If the input was 3D, remove the added channel dimension
#     if input_buffer.ndim == 3:
#         warped = warped.squeeze(0)
    
#     print('Saving warped image to disk...')
#     save_volume(warped.numpy(), field_aff, field_h, output_image)
    
#     print('Warping completed successfully!')
    

# def warp_image(input_image, output_image, field, nearest=False, threads=1):
#     """
#     Warps a 3D or 4D input image using a 4D deformation field and saves the output image.

#     Parameters:
#     - input_image: Path to the input image file.
#     - output_image: Path where the warped image will be saved.
#     - field: Path to the 4D deformation field file.
#     - nearest: Boolean indicating whether to use nearest-neighbor interpolation. Defaults to False (linear).
#     - threads: Number of CPU threads to use. Defaults to 1.
#     """
#     if input_image is None or output_image is None or field is None:
#         raise ValueError("Input image, output image, and field must all be provided")
    
#     # Thread handling
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
#     if threads == 1:
#         print('Using 1 thread')
#     elif threads < 0:
#         threads = os.cpu_count()
#         print(f'Using all available threads ({threads})')
#     else:
#         print(f'Using {threads} threads')
#     torch.set_num_threads(threads)

#     print('Reading deformation field...')
#     field_buffer, field_aff, field_h = load_volume(field, im_only=False, squeeze=True, dtype=None)
    
#     if field_buffer.ndim != 4:
#         raise ValueError('Deformation field must be a 4D array')
#     if field_buffer.shape[3] != 3:
#         raise ValueError('Deformation field must have 3 components in the last dimension')
    
#     print('Reading input image...')
#     input_buffer, input_aff, input_h = load_volume(input_image, im_only=False, squeeze=True, dtype=None)
    
#     # Determine if the input image is 3D or 4D
#     if input_buffer.ndim == 3:
#         input_tensor = torch.tensor(input_buffer, device='cpu', dtype=torch.float32).unsqueeze(0)  # Add channel dimension
#         print('Input image is 3D. Added channel dimension.')
#     elif input_buffer.ndim == 4:
#         input_tensor = torch.tensor(input_buffer, device='cpu', dtype=torch.float32)
#         print('Input image is 4D.')
#     else:
#         raise ValueError('Input image must be either 3D or 4D')

#     print('Preparing deformation field...')
#     affine = torch.tensor(np.linalg.inv(input_aff), device='cpu', dtype=torch.float32)
#     field_tensor = torch.tensor(field_buffer, device='cpu', dtype=torch.float32)
    
#     # Apply affine transformation to the deformation field
#     II = affine[0, 0] * field_tensor[:, :, :, 0] + affine[0, 1] * field_tensor[:, :, :, 1] + affine[0, 2] * field_tensor[:, :, :, 2] + affine[0, 3]
#     JJ = affine[1, 0] * field_tensor[:, :, :, 0] + affine[1, 1] * field_tensor[:, :, :, 1] + affine[1, 2] * field_tensor[:, :, :, 2] + affine[1, 3]
#     KK = affine[2, 0] * field_tensor[:, :, :, 0] + affine[2, 1] * field_tensor[:, :, :, 1] + affine[2, 2] * field_tensor[:, :, :, 2] + affine[2, 3]
    
#     print('Deformation fields (II, JJ, KK) prepared.')

#     print('Performing interpolation...')
#     if nearest:
#         warped = fast_3D_interp_torch(input_tensor, II, JJ, KK, mode='nearest')
#     else:
#         warped = fast_3D_interp_torch(input_tensor, II, JJ, KK, mode='linear')
    
#     # If the input was 3D, remove the added channel dimension
#     if input_buffer.ndim == 3:
#         warped = warped.squeeze(0)
#         print('Removed added channel dimension from warped image.')

#     print('Saving warped image to disk...')
#     save_volume(warped.numpy(), field_aff, field_h, output_image)
    
#     print('Warping completed successfully!')
    

def warp_image(input_image, output_image, field, nearest=False, threads=1):
    """
    Warps a 3D or 4D input image using a 4D deformation field and saves the output image.

    Parameters:
    - input_image (str): Path to the input image file.
    - output_image (str): Path where the warped image will be saved.
    - field (str): Path to the 4D deformation field file.
    - nearest (bool): Whether to use nearest-neighbor interpolation. Defaults to False (linear).
    - threads (int): Number of CPU threads to use. Defaults to 1.
    """
    if input_image is None or output_image is None or field is None:
        raise ValueError("Input image, output image, and field must all be provided")

    # Disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Set number of threads
    if threads == 1:
        print('Using 1 thread')
    elif threads < 0:
        threads = os.cpu_count()
        print(f'Using all available threads ({threads})')
    else:
        print(f'Using {threads} threads')
    torch.set_num_threads(threads)

    print('Reading deformation field...')
    field_buffer, field_aff, field_h = load_volume(field, im_only=False, squeeze=True, dtype=None)

    if field_buffer.ndim != 4:
        raise ValueError('Deformation field must be a 4D array')
    if field_buffer.shape[3] != 3:
        raise ValueError('Deformation field must have 3 components in the last dimension')

    print('Reading input image...')
    input_buffer, input_aff, input_h = load_volume(input_image, im_only=False, squeeze=True, dtype=None)
    print(f'Input image shape: {input_buffer.shape}')

    # Determine if the input image is 3D or 4D
    if input_buffer.ndim == 3:
        # 3D image: add a channel dimension at the beginning
        input_tensor = torch.tensor(input_buffer, device='cpu', dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W, D)
        print('Input image is 3D. Added channel dimension.')
    elif input_buffer.ndim == 4:
        # 4D image: assume channel-last, rearrange to (C, H, W, D)
        input_tensor = torch.tensor(input_buffer, device='cpu', dtype=torch.float32).permute(3, 0, 1, 2)  # Shape: (C, H, W, D)
        print('Input image is 4D. Rearranged to (C, H, W, D).')
    else:
        raise ValueError('Input image must be either 3D or 4D')

    print(f'Input tensor shape after processing: {input_tensor.shape}')

    print('Preparing deformation field...')
    affine = torch.tensor(np.linalg.inv(input_aff), device='cpu', dtype=torch.float32)
    field_tensor = torch.tensor(field_buffer, device='cpu', dtype=torch.float32)

    # Apply affine transformation to the deformation field
    II = affine[0, 0] * field_tensor[:, :, :, 0] + affine[0, 1] * field_tensor[:, :, :, 1] + affine[0, 2] * field_tensor[:, :, :, 2] + affine[0, 3]
    JJ = affine[1, 0] * field_tensor[:, :, :, 0] + affine[1, 1] * field_tensor[:, :, :, 1] + affine[1, 2] * field_tensor[:, :, :, 2] + affine[1, 3]
    KK = affine[2, 0] * field_tensor[:, :, :, 0] + affine[2, 1] * field_tensor[:, :, :, 1] + affine[2, 2] * field_tensor[:, :, :, 2] + affine[2, 3]

    print('Deformation fields (II, JJ, KK) prepared.')
    print(f'Deformation field shapes: II: {II.shape}, JJ: {JJ.shape}, KK: {KK.shape}')

    print('Performing interpolation...')
    if nearest:
        warped = fast_3D_interp_torch(input_tensor, II, JJ, KK, mode='nearest')
    else:
        warped = fast_3D_interp_torch(input_tensor, II, JJ, KK, mode='linear')

    print(f'Warped tensor shape before rearranging: {warped.shape}')

    # If the input was 3D, remove the added channel dimension
    if input_buffer.ndim == 3:
        warped = warped.squeeze(0)  # Shape: (H, W, D)
        print('Removed added channel dimension from warped image.')
    elif input_buffer.ndim == 4:
        # Rearrange back to (H, W, D, C)
        warped = warped.permute(1, 2, 3, 0)  # Shape: (H, W, D, C)
        print('Rearranged warped tensor back to (H, W, D, C).')

    print(f'Warped tensor shape after rearranging: {warped.shape}')

    print('Saving warped image to disk...')
    save_volume(warped.numpy(), field_aff, field_h, output_image)

    print('Warping completed successfully!')

#######################
# Auxiliary functions #
#######################

def load_volume(path_volume, im_only=True, squeeze=True, dtype=None):

    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path):
    mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)

        nib.save(nifty, path)


def mkdir(path_dir):

    if len(path_dir)>0:
        if path_dir[-1] == '/':
            path_dir = path_dir[:-1]
        if not os.path.isdir(path_dir):
            list_dir_to_create = [path_dir]
            while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
                list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
            for dir_to_create in reversed(list_dir_to_create):
                os.mkdir(dir_to_create)



# def fast_3D_interp_torch(X, II, JJ, KK, mode):
#     if mode=='nearest':
#         ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
#         IIr = torch.round(II[ok]).long()
#         JJr = torch.round(JJ[ok]).long()
#         KKr = torch.round(KK[ok]).long()
#         c = X[IIr, JJr, KKr]
#         Y = torch.zeros(II.shape, device='cpu')
#         Y[ok] = c.float()
        
#     elif mode=='linear':
#         ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
#         IIv = II[ok]
#         JJv = JJ[ok]
#         KKv = KK[ok]

#         fx = torch.floor(IIv).long()
#         cx = fx + 1
#         cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
#         wcx = IIv - fx
#         wfx = 1 - wcx

#         fy = torch.floor(JJv).long()
#         cy = fy + 1
#         cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
#         wcy = JJv - fy
#         wfy = 1 - wcy

#         fz = torch.floor(KKv).long()
#         cz = fz + 1
#         cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
#         wcz = KKv - fz
#         wfz = 1 - wcz

#         c000 = X[fx, fy, fz]
#         c100 = X[cx, fy, fz]
#         c010 = X[fx, cy, fz]
#         c110 = X[cx, cy, fz]
#         c001 = X[fx, fy, cz]
#         c101 = X[cx, fy, cz]
#         c011 = X[fx, cy, cz]
#         c111 = X[cx, cy, cz]

#         c00 = c000 * wfx + c100 * wcx
#         c01 = c001 * wfx + c101 * wcx
#         c10 = c010 * wfx + c110 * wcx
#         c11 = c011 * wfx + c111 * wcx

#         c0 = c00 * wfy + c10 * wcy
#         c1 = c01 * wfy + c11 * wcy

#         c = c0 * wfz + c1 * wcz

#         Y = torch.zeros(II.shape, device='cpu')
#         Y[ok] = c.float()

#     else:
#         print('mode must be linear or nearest')

#     return Y
                
# def fast_3D_interp_torch(X, II, JJ, KK, mode='linear'):
#     """
#     Performs fast 3D interpolation on input tensor X using the provided deformation fields II, JJ, KK.

#     Parameters:
#     - X: Input tensor of shape (C, H, W, D) or (1, H, W, D) for single-channel.
#     - II, JJ, KK: Deformation fields corresponding to the x, y, z coordinates.
#     - mode: Interpolation mode - 'linear' or 'nearest'. Defaults to 'linear'.

#     Returns:
#     - Y: Warped tensor with the same shape as the input spatial dimensions.
#     """
#     if X.ndim != 4:
#         raise ValueError('Input tensor X must be 4D (C, H, W, D)')
    
#     C, H, W, D = X.shape
#     device = X.device
#     dtype = X.dtype
    
#     # Initialize the output tensor
#     Y = torch.zeros((C, II.shape[0], II.shape[1], II.shape[2]), device=device, dtype=dtype)
    
#     print(f'Interpolating {C} channel(s) with mode="{mode}".')

#     # Process each channel separately
#     for channel in range(C):
#         Xc = X[channel]
        
#         if mode == 'nearest':
#             ok = (II >= 0) & (JJ >= 0) & (KK >= 0) & \
#                  (II <= H - 1) & (JJ <= W - 1) & (KK <= D - 1)
#             IIr = torch.round(II[ok]).long()
#             JJr = torch.round(JJ[ok]).long()
#             KKr = torch.round(KK[ok]).long()
            
#             # Debugging statements
#             print(f'Channel {channel}: {ok.sum().item()} valid points for nearest interpolation.')
#             print(f'IIr max: {IIr.max().item()}, IIr min: {IIr.min().item()}')
#             print(f'JJr max: {JJr.max().item()}, JJr min: {JJr.min().item()}')
#             print(f'KKr max: {KKr.max().item()}, KKr min: {KKr.min().item()}')
            
#             c_vals = Xc[IIr, JJr, KKr]
#             Y[channel][ok] = c_vals.float()
        
#         elif mode == 'linear':
#             ok = (II >= 0) & (JJ >= 0) & (KK >= 0) & \
#                  (II <= H - 1) & (JJ <= W - 1) & (KK <= D - 1)
#             IIv = II[ok]
#             JJv = JJ[ok]
#             KKv = KK[ok]
    
#             fx = torch.floor(IIv).long()
#             cx = fx + 1
#             cx = torch.clamp(cx, max=H-1)
#             wcx = IIv - fx
#             wfx = 1.0 - wcx
    
#             fy = torch.floor(JJv).long()
#             cy = fy + 1
#             cy = torch.clamp(cy, max=W-1)
#             wcy = JJv - fy
#             wfy = 1.0 - wcy
    
#             fz = torch.floor(KKv).long()
#             cz = fz + 1
#             cz = torch.clamp(cz, max=D-1)
#             wcz = KKv - fz
#             wfz = 1.0 - wcz
    
#             # Debugging statements
#             print(f'Channel {channel}: {ok.sum().item()} valid points for linear interpolation.')
#             print(f'fx max: {fx.max().item()}, fx min: {fx.min().item()}')
#             print(f'cx max: {cx.max().item()}, cx min: {cx.min().item()}')
#             print(f'fy max: {fy.max().item()}, fy min: {fy.min().item()}')
#             print(f'cy max: {cy.max().item()}, cy min: {cy.min().item()}')
#             print(f'fz max: {fz.max().item()}, fz min: {fz.min().item()}')
#             print(f'cz max: {cz.max().item()}, cz min: {cz.min().item()}')
    
#             # Gather voxel values
#             try:
#                 c000 = Xc[fx, fy, fz]
#                 c100 = Xc[cx, fy, fz]
#                 c010 = Xc[fx, cy, fz]
#                 c110 = Xc[cx, cy, fz]
#                 c001 = Xc[fx, fy, cz]
#                 c101 = Xc[cx, fy, cz]
#                 c011 = Xc[fx, cy, cz]
#                 c111 = Xc[cx, cy, cz]
#             except IndexError as e:
#                 print(f'IndexError while gathering voxel values in channel {channel}: {e}')
#                 raise
    
#             # Trilinear interpolation
#             c00 = c000 * wfx + c100 * wcx
#             c01 = c001 * wfx + c101 * wcx
#             c10 = c010 * wfx + c110 * wcx
#             c11 = c011 * wfx + c111 * wcx
    
#             c0 = c00 * wfy + c10 * wcy
#             c1 = c01 * wfy + c11 * wcy
    
#             c_interp = c0 * wfz + c1 * wcz
    
#             Y[channel][ok] = c_interp.float()
        
#         else:
#             raise ValueError("Interpolation mode must be 'linear' or 'nearest'")
    
#     print('Interpolation completed.')
#     return Y
                
def fast_3D_interp_torch(X, II, JJ, KK, mode='linear'):
    """
    Performs fast 3D interpolation on input tensor X using the provided deformation fields II, JJ, KK.

    Parameters:
    - X (torch.Tensor): Input tensor of shape (C, H, W, D).
    - II (torch.Tensor): Deformation field for the x-axis (H, W, D).
    - JJ (torch.Tensor): Deformation field for the y-axis (H, W, D).
    - KK (torch.Tensor): Deformation field for the z-axis (H, W, D).
    - mode (str): Interpolation mode - 'linear' or 'nearest'. Defaults to 'linear'.

    Returns:
    - torch.Tensor: Warped tensor with shape (C, H, W, D).
    """
    if X.ndim != 4:
        raise ValueError('Input tensor X must be 4D (C, H, W, D)')

    C, H, W, D = X.shape
    device = X.device
    dtype = X.dtype

    # Initialize the output tensor
    Y = torch.zeros((C, II.shape[0], II.shape[1], II.shape[2]), device=device, dtype=dtype)

    print(f'Interpolating {C} channel(s) with mode="{mode}".')
    print(f'Output tensor shape: {Y.shape}')

    # Process each channel separately
    for channel in range(C):
        Xc = X[channel]

        if mode == 'nearest':
            # Nearest-neighbor interpolation
            valid_mask = (II >= 0) & (JJ >= 0) & (KK >= 0) & \
                         (II <= H - 1) & (JJ <= W - 1) & (KK <= D - 1)
            IIr = torch.round(II[valid_mask]).long()
            JJr = torch.round(JJ[valid_mask]).long()
            KKr = torch.round(KK[valid_mask]).long()

            # Debugging statements
            print(f'Channel {channel + 1}/{C}: {valid_mask.sum().item()} valid points for nearest interpolation.')
            print(f'IIr max: {IIr.max().item()}, IIr min: {IIr.min().item()}')
            print(f'JJr max: {JJr.max().item()}, JJr min: {JJr.min().item()}')
            print(f'KKr max: {KKr.max().item()}, KKr min: {KKr.min().item()}')

            if IIr.numel() > 0:
                c_vals = Xc[IIr, JJr, KKr]
                Y[channel][valid_mask] = c_vals.float()
            else:
                print(f'Channel {channel + 1}: No valid points for nearest interpolation.')

        elif mode == 'linear':
            # Trilinear interpolation
            valid_mask = (II >= 0) & (JJ >= 0) & (KK >= 0) & \
                         (II <= H - 1) & (JJ <= W - 1) & (KK <= D - 1)
            IIv = II[valid_mask]
            JJv = JJ[valid_mask]
            KKv = KK[valid_mask]

            # Floor and ceil for interpolation
            fx = torch.floor(IIv).long()
            fy = torch.floor(JJv).long()
            fz = torch.floor(KKv).long()

            cx = fx + 1
            cy = fy + 1
            cz = fz + 1

            # Clamp to ensure indices are within bounds
            cx = torch.clamp(cx, max=H-1)
            cy = torch.clamp(cy, max=W-1)
            cz = torch.clamp(cz, max=D-1)

            # Weights
            wfx = 1.0 - (IIv - fx.float())
            wfy = 1.0 - (JJv - fy.float())
            wfz = 1.0 - (KKv - fz.float())

            wcx = IIv - fx.float()
            wcy = JJv - fy.float()
            wcz = KKv - fz.float()

            # Debugging statements
            print(f'Channel {channel + 1}/{C}: {valid_mask.sum().item()} valid points for linear interpolation.')
            print(f'fx max: {fx.max().item()}, fx min: {fx.min().item()}')
            print(f'cx max: {cx.max().item()}, cx min: {cx.min().item()}')
            print(f'fy max: {fy.max().item()}, fy min: {fy.min().item()}')
            print(f'cy max: {cy.max().item()}, cy min: {cy.min().item()}')
            print(f'fz max: {fz.max().item()}, fz min: {fz.min().item()}')
            print(f'cz max: {cz.max().item()}, cz min: {cz.min().item()}')

            try:
                # Gather voxel values for the 8 surrounding points
                c000 = Xc[fx, fy, fz]
                c100 = Xc[cx, fy, fz]
                c010 = Xc[fx, cy, fz]
                c110 = Xc[cx, cy, fz]
                c001 = Xc[fx, fy, cz]
                c101 = Xc[cx, fy, cz]
                c011 = Xc[fx, cy, cz]
                c111 = Xc[cx, cy, cz]
            except IndexError as e:
                print(f'IndexError while gathering voxel values in channel {channel + 1}: {e}')
                raise

            # Perform trilinear interpolation
            c00 = c000 * wfx + c100 * wcx
            c10 = c010 * wfx + c110 * wcx
            c01 = c001 * wfx + c101 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c_interp = c0 * wfz + c1 * wcz

            if c_interp.numel() > 0:
                Y[channel][valid_mask] = c_interp.float()
            else:
                print(f'Channel {channel + 1}: No valid points for linear interpolation.')

        else:
            raise ValueError("Interpolation mode must be 'linear' or 'nearest'")

    print('Interpolation completed.')
    return Y

# execute script
if __name__ == '__main__':
    main()
import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
import random
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
import argparse
import time
import gc
import psutil

def ensure_float32_tensor(x, device):
    """Ensure the input is a float32 tensor"""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    return x.to(dtype=torch.float32, device=device)

def create_spatial_g(size, device):
    """Create a spatially varying anisotropy factor field - fully vectorized version"""
    H, W = size
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    g_field = torch.zeros((H, W), device=device)
    num_centers = random.randint(3, 5)
    

    centers = []
    for _ in range(num_centers):
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        value = random.uniform(0.5, 0.95)
        sigma = random.uniform(min(H, W)//4, min(H, W)//2)
        centers.append((x, y, value, sigma))
    

    for x, y, value, sigma in centers:
        dist = torch.sqrt((X - x)**2 + (Y - y)**2)
        g_field += value * torch.exp(-dist**2 / (2*sigma**2))
    

    g_min, g_max = g_field.min(), g_field.max()
    g_field = 0.5 + 0.45 * (g_field - g_min) / (g_max - g_min + 1e-8)
    
    return g_field

def create_nonuniform_lighting(size, device):
    """Create a non-uniform lighting field - fully vectorized version"""
    H, W = size
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    light_map = torch.zeros((H, W), device=device)
    num_sources = random.randint(2, 3)
    

    sources = []
    for _ in range(num_sources):
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        intensity = random.uniform(0.8, 1.5)
        sigma = random.uniform(W//4, W//2)
        sources.append((x, y, intensity, sigma))
    

    for x, y, intensity, sigma in sources:
        dist = torch.sqrt((X - x)**2 + (Y - y)**2)
        light_map += intensity * torch.exp(-dist**2 / (2*sigma**2))
    

    light_map = 0.5 + light_map / (light_map.max() + 1e-8)
    return torch.clamp(light_map, 0.5, 1.5)

def generate_psf_kernels(depth_samples, beta_samples, g_samples, kernel_size=33, device='cuda'):
    """
    Batch generate PSF kernels - utilize GPU parallelism

    Args:
    - depth_samples: batch depth values [N]
    - beta_samples: batch beta values [N]
    - g_samples: batch g values [N]
    - kernel_size: PSF kernel size

    Returns:
    - kernels: PSF kernels tensor of shape [N, kernel_size, kernel_size]
    """
    batch_size = len(depth_samples)
    

    center = kernel_size // 2
    Y, X = torch.meshgrid(
        torch.arange(-center, center + 1, device=device),
        torch.arange(-center, center + 1, device=device),
        indexing='ij'
    )
    

    r = torch.sqrt(X**2 + Y**2 + 1e-6).unsqueeze(0)
    

    depths = depth_samples.view(-1, 1, 1)
    betas = beta_samples.view(-1, 1, 1)
    g_values = g_samples.view(-1, 1, 1)
    

    cos_theta = depths / torch.sqrt(r**2 + depths**2 + 1e-6)
    

    g_squared = g_values**2
    

    phase = (1 - g_squared) / (4 * np.pi * (1 + g_squared - 2 * g_values * cos_theta)**1.5)
    

    transmission = torch.exp(-betas * depths)
    

    kernels = phase * transmission / (r**2 + 1e-6)
    

    for i in range(batch_size):
        kernels[i] = kernels[i] / (kernels[i].sum() + 1e-6)
    
    return kernels

def efficient_batchwise_convolution(img, depth, beta_D, g_field, device):
    """
    Efficient batchwise convolution implementation - zero padding at boundaries instead of asymmetric padding
    """
    H, W, C = img.shape
    

    img_tensor = torch.zeros((C, H, W), device=device)
    for c in range(C):
        img_tensor[c] = torch.from_numpy(img[:, :, c]).to(device)
    

    depth = ensure_float32_tensor(depth, device)
    beta_D = ensure_float32_tensor(beta_D, device)
    g_field = ensure_float32_tensor(g_field, device)
    

    kernel_size = 33
    pad = kernel_size // 2
    

    result = torch.zeros_like(img_tensor)
    weight = torch.zeros_like(img_tensor)

    grid_step = 32
    

    y_samples = list(range(0, H, grid_step))
    if y_samples[-1] < H-1:
        y_samples.append(H-1)
    
    x_samples = list(range(0, W, grid_step))
    if x_samples[-1] < W-1:
        x_samples.append(W-1)
    

    for c in range(C):
        print(f"Processing channel {c+1}/{C}...")
        

        img_padded = F.pad(img_tensor[c].unsqueeze(0).unsqueeze(0), 
                           (pad, pad, pad, pad), 
                           mode='constant', value=0)
        

        depth_samples = []
        beta_samples = []
        g_samples = []
        coords = []
        
        for y in y_samples:
            for x in x_samples:
                depth_samples.append(depth[y, x])
                beta_samples.append(beta_D[y, x, c])
                g_samples.append(g_field[y, x])
                coords.append((y, x))
        

        depth_samples = torch.tensor(depth_samples, device=device)
        beta_samples = torch.tensor(beta_samples, device=device)
        g_samples = torch.tensor(g_samples, device=device)
        
        print(f"Batch generating {len(depth_samples)} PSF kernels for channel {c+1}/{C}...")
        psf_kernels = generate_psf_kernels(depth_samples, beta_samples, g_samples, kernel_size, device)
        

        print(f"Applying convolution for channel {c+1}/{C}...")
        for idx, (y, x) in enumerate(tqdm(coords)):

            overlap = grid_step // 4
            y_start = max(0, y - grid_step//2 - overlap)
            y_end = min(H, y + grid_step//2 + overlap)
            x_start = max(0, x - grid_step//2 - overlap)
            x_end = min(W, x + grid_step//2 + overlap)
            

            block_img = img_padded[0, 0, 
                                  y_start:y_end+2*pad, 
                                  x_start:x_end+2*pad]
            

            kernel = psf_kernels[idx].unsqueeze(0).unsqueeze(0)
            block_result = F.conv2d(
                block_img.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=0
            )
            

            block_h, block_w = y_end-y_start, x_end-x_start
            yy = torch.arange(block_h, device=device).float() - (y - y_start)
            xx = torch.arange(block_w, device=device).float() - (x - x_start)
            yy = yy.view(-1, 1).expand(-1, block_w)
            xx = xx.view(1, -1).expand(block_h, -1)
            dist = torch.sqrt(yy**2 + xx**2) / (grid_step/2)
            weight_block = torch.exp(-dist**2)
            

            result[c, y_start:y_end, x_start:x_end] += \
                block_result[0, 0] * weight_block
            weight[c, y_start:y_end, x_start:x_end] += weight_block
    

    result = torch.where(weight > 0, result / (weight + 1e-6), result)
    

    result_np = torch.zeros((H, W, C), device=device)
    for c in range(C):
        result_np[:, :, c] = result[c]
    
    return result_np

def generate_degraded_image(img, depth, coef_data, gpu_id=0):
    """Main function to generate degraded image on a single GPU"""
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        start_time = time.time()
        

        B_inf = torch.tensor(coef_data['B_inf'], dtype=torch.float32, device=device)
        beta_B = torch.tensor(coef_data['beta_B'], dtype=torch.float32, device=device)
        

        img = ensure_float32_tensor(img, device)
        depth = ensure_float32_tensor(depth, device)
        

        beta_D = torch.zeros((depth.shape[0], depth.shape[1], 3), device=device)
        for c, coef in enumerate([coef_data['Dcoefs_r'], coef_data['Dcoefs_g'], coef_data['Dcoefs_b']]):
            a, b, c_val, d = coef
            beta_D[..., c] = (a * torch.exp(b * depth)) + (c_val * torch.exp(d * depth))
        beta_D *= 0.5  
        

        print(f"GPU {gpu_id}: Generating spatially varying g field and lighting...")
        g_field = create_spatial_g(depth.shape, device)
        lighting = create_nonuniform_lighting(depth.shape, device)
        

        img = img * lighting.unsqueeze(2)
        

        print(f"GPU {gpu_id}: Applying spatially varying convolution...")
        direct = efficient_batchwise_convolution(img.cpu().numpy(), depth, beta_D, g_field, device)
        

        print(f"GPU {gpu_id}: Calculating final degraded image...")
        t = torch.exp(-beta_D * depth.unsqueeze(2))
        B_term = B_inf * (1 - torch.exp(-beta_B * depth.unsqueeze(2)))
        degraded = direct * t + B_term
        

        degraded = torch.clamp(degraded, 0.0, 1.0)
        
        print(f"GPU {gpu_id}: Degraded image generated, time taken: {time.time() - start_time:.2f}s")
        return degraded, lighting, g_field, beta_D
        
    except Exception as e:
        print(f"Error processing image on GPU {gpu_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def worker_process_batch(rank, img_files, batch_data, coefs, out_dir, params_dir, checkpoint_file, return_dict):
    """Worker process optimized for batch mode"""
    try:

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        print(f"Process {rank} started on GPU {rank}")
        

        processed_count = 0
        
        for img_f in img_files:
            start_time = time.time()
            print(f"GPU {rank}: Start processing image {img_f}")
            

            img, depth = batch_data[img_f]
            

            k = random.sample(list(coefs.keys()), 1)[0]
            selected_coefs = coefs[k]
            

            coef_data = {
                'B_inf': [
                    selected_coefs["Bcoefs_r"][0],
                    selected_coefs["Bcoefs_g"][0],
                    selected_coefs["Bcoefs_b"][0]
                ],
                'beta_B': [
                    selected_coefs["Bcoefs_r"][1],
                    selected_coefs["Bcoefs_g"][1],
                    selected_coefs["Bcoefs_b"][1]
                ],
                'Dcoefs_r': selected_coefs["Dcoefs_r"],
                'Dcoefs_g': selected_coefs["Dcoefs_g"],
                'Dcoefs_b': selected_coefs["Dcoefs_b"]
            }
            

            result = generate_degraded_image(img, depth, coef_data, gpu_id=rank)
            
            if result is not None:
                degraded, lighting, g_field, beta_D = result
                

                print(f"GPU {rank}: Post-processing...")
                degraded_cpu = degraded.cpu().numpy()
                

                del degraded
                torch.cuda.empty_cache()
                

                sigma_est = estimate_sigma(degraded_cpu, multichannel=True, average_sigmas=True) / 10.0
                degraded_denoised = denoise_tv_chambolle(degraded_cpu, sigma_est, multichannel=True)
                

                print(f"GPU {rank}: Saving results...")
                cv2.imwrite(
                    os.path.join(out_dir, img_f),
                    cv2.cvtColor((np.clip(degraded_denoised * 255, 0, 255)).astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                

                def safe_to_numpy(tensor_or_array):
                    if isinstance(tensor_or_array, torch.Tensor):
                        return tensor_or_array.cpu().numpy()
                    return tensor_or_array
                

                params = {
                    'depth': safe_to_numpy(depth),
                    'beta_D': safe_to_numpy(beta_D),
                    'beta_B': coef_data['beta_B'],
                    'g': safe_to_numpy(g_field),
                    'lighting': safe_to_numpy(lighting),
                    'B_inf': coef_data['B_inf']
                }
                np.save(os.path.join(params_dir, f"{os.path.splitext(img_f)[0]}.npy"), params)
                

                with open(checkpoint_file, 'a') as f:
                    f.write(f"{img_f}\n")
                
                processed_count += 1
                total_time = time.time() - start_time
                print(f"GPU {rank}: Successfully processed image {img_f}, time taken: {total_time:.2f}s")
            

            torch.cuda.empty_cache()
            gc.collect()
        
        return_dict[rank] = processed_count
        print(f"Process {rank} finished, processed {processed_count} images")
    
    except Exception as e:
        print(f"Process {rank} encountered an error: {str(e)}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = 0

def preload_batch(batch_files, image_dir, depth_dir):
    """Load only a batch of images into memory instead of all images"""
    print(f"Preloading {len(batch_files)} images...")
    result = {}
    for img_f in tqdm(batch_files):
        try:

            img = cv2.cvtColor(cv2.imread(os.path.join(image_dir, img_f)), cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0
            

            depth = cv2.imread(os.path.join(depth_dir, img_f), cv2.IMREAD_GRAYSCALE)
            if depth is None or img.shape[:2] != depth.shape:
                continue
                

            depth = cv2.GaussianBlur(depth, (7, 7), 5) * 1.0
            if depth.max() == depth.min():
                continue
            depth = np.nan_to_num((depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)).astype(np.float32)
            depth = 10. * depth + 2.
            
            result[img_f] = (img, depth)
        except Exception as e:
            print(f"Error preloading image {img_f}: {str(e)}")
    
    print(f"Successfully preloaded {len(result)}/{len(batch_files)} images")
    return result

def distribute_files(files, num_processes):
    """Distribute files evenly among multiple processes"""
    result = [[] for _ in range(num_processes)]
    for i, f in enumerate(files):
        result[i % num_processes].append(f)
    return result

def batch_processor(batch_files, image_dir, depth_dir, out_dir, params_dir, coefs, gpu_ids, checkpoint_file):
    """Process a batch of image files"""

    batch_data = preload_batch(batch_files, image_dir, depth_dir)
    if not batch_data:
        return 0
    

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    

    distributed_files = distribute_files(list(batch_data.keys()), len(gpu_ids))
    

    processes = []
    for i, rank in enumerate(gpu_ids):
        if i < len(distributed_files) and distributed_files[i]:  
            p = mp.Process(
                target=worker_process_batch,
                args=(rank, distributed_files[i], batch_data, coefs, out_dir, params_dir, checkpoint_file, return_dict)
            )
            p.start()
            processes.append(p)
    

    for p in processes:
        p.join()
    

    processed_count = sum(return_dict.values())
    

    del batch_data
    gc.collect()
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='Batch-optimized underwater image simulation degradation')
    parser.add_argument('--image_dir', default=None, help='Input image directory')
    parser.add_argument('--depth_dir', default=None, help='Input depth map directory')
    parser.add_argument('--out_dir', default=None, help='Output directory')
    parser.add_argument('--coeff_file', default='coeffs.json', help='Coefficient file path')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images per batch')
    args = parser.parse_args()
    
    # Set paths
    image_dir = args.image_dir
    depth_dir = args.depth_dir
    out_dir = args.out_dir if args.out_dir else f"output/synthetic_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    params_dir = os.path.join(out_dir, 'params')
    checkpoint_file = os.path.join(out_dir, 'checkpoint.txt')

    # Create directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    # Check available GPU count
    available_gpus = min(args.num_gpus, torch.cuda.device_count())
    gpu_ids = list(range(available_gpus))
    print(f"Using {available_gpus} GPUs for processing")

    # Load coefficient file
    if not os.path.exists(args.coeff_file):
        print(f"{args.coeff_file} file does not exist")
        return
    coefs = json.load(open(args.coeff_file, 'r'))

    # Get image file list
    img_files = sorted(os.listdir(image_dir))

    # Check for checkpoint
    processed_files = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_files = set(f.read().splitlines())
        print(f"Resuming from checkpoint: {len(processed_files)} files already processed")

    # Filter out already processed files
    img_files = [f for f in img_files if f not in processed_files]

    if not img_files:
        print("All images have been processed")
        return

    print(f"About to process {len(img_files)} images...")

    # Calculate batch size - auto adjust based on available memory
    mem = psutil.virtual_memory()
    available_mem_gb = mem.available / (1024**3)
    print(f"Available system memory: {available_mem_gb:.2f} GB")

    # Estimate per image size (assuming 1080p images)
    est_img_size_gb = 0.03  # ~30MB per image + depth map

    # Calculate safe batch size (use 30% of available memory)
    safe_batch_size = int((available_mem_gb * 0.3) / est_img_size_gb)
    batch_size = min(args.batch_size, safe_batch_size)
    batch_size = max(batch_size, 1)  # Ensure at least 1 image per batch

    print(f"Auto-adjusted batch size: {batch_size} images")

    # Process images in batches
    total_processed = 0
    for i in range(0, len(img_files), batch_size):
        batch = img_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(img_files) + batch_size - 1)//batch_size}, containing {len(batch)} images")

        # Process this batch
        processed = batch_processor(batch, image_dir, depth_dir, out_dir, params_dir, coefs, gpu_ids, checkpoint_file)
        total_processed += processed

        # Force memory cleanup
        gc.collect()

        print(f"Total processed: {total_processed}/{len(img_files)} images")

    print("All images processed!")

if __name__ == '__main__':
    main()

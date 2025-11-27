import os
from pathlib import Path
import PIL.Image
from typing import List
import click
import numpy as np
import torch
from tqdm import tqdm
import gc

import legacy
import dnnlib
from torch_utils import gen_utils
from gen_images import parse_range

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=42)
@click.option('--centroids-path', type=str, help='Pass path to precomputed centroids to enable multimodal truncation')
@click.option('--classes', type=parse_range, help='List of classes (e.g., \'0,1,4-6\')', required=True)
@click.option('--samples-per-class', help='Samples per class.', type=int, default=4)
@click.option('--grid-width', help='Total width of image grid', type=int, default=32)
@click.option('--batch-gpu', help='Samples per pass for image generation, adapt to fit on GPU', type=int, default=32)
@click.option('--batch-latent', help='Batch size for latent generation (smaller = less memory, default: same as batch-gpu)', type=int, default=None)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
def generate_samplesheet(
    network_pkl: str,
    truncation_psi: float,
    seed: int,
    centroids_path: str,
    classes: List[int],
    samples_per_class: int,
    batch_gpu: int,
    batch_latent: int,
    grid_width: int,
    outdir: str,
    desc: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).requires_grad_(False)

    # setup
    os.makedirs(outdir, exist_ok=True)
    desc_full = f'{Path(network_pkl).stem}_trunc_{truncation_psi}'
    if desc is not None: desc_full += f'-{desc}'
    run_dir = Path(gen_utils.make_run_dir(outdir, desc_full))

    # Use batch_latent if provided, otherwise use batch_gpu
    if batch_latent is None:
        batch_latent = batch_gpu

    print(f'Generating {samples_per_class} samples per class using batch sizes: latent={batch_latent}, image={batch_gpu}')
    
    # Process each class separately to avoid memory issues
    for class_idx in classes:
        # Create class folder
        class_dir = run_dir / f'class_{class_idx}'
        class_dir.mkdir(exist_ok=True)
        
        # Calculate number of batches needed
        num_batches = (samples_per_class + batch_latent - 1) // batch_latent
        image_counter = 0
        
        # Use a separate random state for this class to ensure reproducibility
        class_seed = seed + class_idx if seed is not None else None
        
        print(f'Processing class {class_idx}...')
        for batch_idx in tqdm(range(num_batches), desc=f'Class {class_idx}'):
            # Calculate how many samples in this batch
            batch_start = batch_idx * batch_latent
            batch_end = min(batch_start + batch_latent, samples_per_class)
            current_batch_size = batch_end - batch_start
            
            # Generate latents for this batch
            # Use different seeds for each batch to ensure variety
            batch_seed = class_seed + batch_idx if class_seed is not None else None
            w = gen_utils.get_w_from_seed(G, current_batch_size, device, truncation_psi, 
                                        seed=batch_seed, centroids_path=centroids_path, 
                                        class_idx=class_idx)
            
            # Generate images in smaller chunks if needed
            # Split w tensor along batch dimension (dim 0) if it's larger than batch_gpu
            if w.shape[0] > batch_gpu:
                w_chunks = w.split(batch_gpu, dim=0)
            else:
                w_chunks = [w]
            
            for w_chunk in w_chunks:
                img = gen_utils.w_to_img(G, w_chunk, to_np=True)
                
                # Save images immediately
                for img_array in img:
                    img_pil = PIL.Image.fromarray(img_array, 'RGB')
                    img_pil.save(class_dir / f'image_{image_counter:06d}.png')
                    image_counter += 1
                
                # Clear GPU memory after each chunk
                del img
                torch.cuda.empty_cache()
            
            # Clear GPU memory after processing all chunks
            del w
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f'Class {class_idx}: Generated {image_counter} images')

if __name__ == "__main__":
    generate_samplesheet()

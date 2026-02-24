import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.utils.data as torchdata
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

import mae_model_brdf
import merlDB.database as db
import neural_data as nd
from merlDB.database import writebin, readbin
from merlFunctions import readMERLBRDF
from merlDB.rendering import Renderer


def score_continuite(brdf_ref):
    """Calculate a continuous score for a BRDF on every angle, not just on the 4D grid. This is done by rendering the BRDF and comparing it to a reference render.

    Args:
        brdf_ref (array): The reference BRDF to score [3,180,90,90]
    """
    # Use L2 norm (RMS) for gradients along each axis
    g1 = torch.sqrt(torch.mean((brdf_ref[:, 1:, :, :] - brdf_ref[:, :-1, :, :]) ** 2))

    # Gradient axe 2 (90)
    g2 = torch.sqrt(torch.mean((brdf_ref[:, :, 1:, :] - brdf_ref[:, :, :-1, :]) ** 2))

    # Gradient axe 3 (90)
    g3 = torch.sqrt(torch.mean((brdf_ref[:, :, :, 1:] - brdf_ref[:, :, :, :-1]) ** 2))

    # print
    print(f"g1: {2*g1.item():.6f}, g2: {g2.item():.6f}, g3: {g3.item():.6f}")

    score = (2*g1 + g2 + g3) / 4

    return score
    


def calculate_psnr(reference, test, max_i=1.0):
    """Calculate PSNR between two arrays"""
    mse = np.mean((reference - test) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_i ** 2) / mse)


def calculate_ssim(reference, test, data_range=None, K1=0.01, K2=0.03, win_size=11, sigma=1.5):
    """
    Calculate SSIM (Structural Similarity Index) between two arrays.
    
    Args:
        reference: Reference array
        test: Test array (same shape as reference)
        data_range: Data range of the input arrays (max_i - min_i). If None, uses max of reference.
        K1, K2: Algorithm parameters (default: 0.01, 0.03)
        win_size: Window size for Gaussian filter (default: 11)
        sigma: Standard deviation for Gaussian filter (default: 1.5)
    
    Returns:
        SSIM value between 0 and 1 (1 means identical)
    """
    if data_range is None:
        data_range = np.max(reference) - np.min(reference)
    
    if data_range == 0:
        return 1.0
    
    # Constants for stability
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # Convert to float64 for precision
    reference = reference.astype(np.float64)
    test = test.astype(np.float64)
    
    # Calculate local means using Gaussian filter
    mu1 = gaussian_filter(reference, sigma=sigma, truncate=win_size/2/sigma)
    mu2 = gaussian_filter(test, sigma=sigma, truncate=win_size/2/sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate local variances and covariance
    sigma1_sq = gaussian_filter(reference ** 2, sigma=sigma, truncate=win_size/2/sigma) - mu1_sq
    sigma2_sq = gaussian_filter(test ** 2, sigma=sigma, truncate=win_size/2/sigma) - mu2_sq
    sigma12 = gaussian_filter(reference * test, sigma=sigma, truncate=win_size/2/sigma) - mu1_mu2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    # Return mean SSIM
    return np.mean(ssim_map)


def render_sphere_multiple_angles(brdf_function, material_name, renderer, angles_list=None, size=400):
    """
    Render a sphere with BRDF from multiple viewing angles.
    
    Args:
        brdf_function: BRDF function to apply
        material_name: Name of the material
        renderer: Renderer object
        angles_list: List of (sphere_idx, tour_idx) tuples for different angles
                     If None, uses default angles
        size: Size of rendered images
    
    Returns:
        List of rendered images (numpy arrays)
    """
    if angles_list is None:
        # Default angles: front, side, top, and some oblique views
        angles_list = [
            (0, 0),   # Front
            (0, 2),   # 72 degrees rotation
            (0, 4),   # 144 degrees rotation  
            (0, 6),   # 216 degrees rotation
            (0, 8),   # 288 degrees rotation
        ]
    
    images = []
    for sphere_idx, tour_idx in angles_list:
        img = renderer.sub_render(
            sphere_idx, 
            tour_idx, 
            brdf_function, 
            name=material_name, 
            save=False, 
            grey_levels=False
        )
        images.append(img)
    
    return images


def display_multi_angle_comparison(images_ref, images_test, material_name, angles_list=None):
    """
    Display multiple angle views of reference and predicted spheres side by side.
    
    Args:
        images_ref: List of reference images
        images_test: List of test images
        material_name: Name of the material
        angles_list: List of angle descriptions (optional)
    """
    n_angles = len(images_ref)
    
    if angles_list is None:
        angles_list = [f"Angle {i+1}" for i in range(n_angles)]
    
    # Create figure with 2 rows (ref, test) and n_angles columns
    fig, axes = plt.subplots(2, n_angles, figsize=(4*n_angles, 8))
    
    # Handle single column case
    if n_angles == 1:
        axes = axes.reshape(2, 1)
    
    for i, (img_ref, img_test) in enumerate(zip(images_ref, images_test)):
        # Reference on top row
        axes[0, i].imshow(Image.fromarray((255*img_ref).astype(np.uint8)))
        if i == 0:
            axes[0, i].set_ylabel("Reference", fontsize=12, fontweight='bold')
        axes[0, i].set_title(angles_list[i] if i < len(angles_list) else f"View {i+1}")
        axes[0, i].axis('off')
        
        # Predicted on bottom row
        axes[1, i].imshow(Image.fromarray((255*img_test).astype(np.uint8)))
        if i == 0:
            axes[1, i].set_ylabel("Predicted", fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
    
    fig.suptitle(f'Multi-Angle BRDF Comparison - {material_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def psnr(materiaux_res="pink-fabric", show_render=False, show_multi_angle=True, max_i=1.0):
    poids = 2 
    dim_latent = 12
    norme = 2
    merldir = "./merlDB/db/brdfs/"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Charger le modèle MAE
    mae = mae_model_brdf.MaskedAutoencoderViT3D(poids=poids, embed_dim=dim_latent, norme=norme)
    checkpoint = torch.load("./../checkpoint_epoch_999_avec_continuite.pt")
    mae.load_state_dict(checkpoint['model_state_dict'])
    mae.to(device)
    mae.eval()
    medians = db.readbin('./merlDB/db/merl_median.binary')

    def median_mapping(albedos, epsilon=0.002):
        return np.log((albedos + epsilon)/(medians + epsilon) + 1)

    def median_unmapping(mapped_albedos, epsilon=0.002):
        return (np.exp(mapped_albedos) -1) * (medians + epsilon)  - epsilon
    
    # Create results/prediction directory if it doesn't exist
    os.makedirs("./results/prediction", exist_ok=True)

    save_dbuilder = db.DBuilder(db_path=merldir, albedo_mapping=median_mapping)
    save_dbuilder.load_mat(materiaux_res)
    save_dataset = nd.DBridge(save_dbuilder, device)
    del save_dbuilder
    for i,data in enumerate(torchdata.DataLoader(save_dataset)):
            X = data['values']
            _, prediction, _ , latent = mae(X, mask_ratio=0.0, ispermute=False)
            prediction = mae.unpatchify(prediction)
            Y = torch.swapaxes(prediction.squeeze(), 0, -1).detach().cpu().numpy()
            Y = median_unmapping(Y)
            writebin(f"./results/prediction/{materiaux_res}-pred.binary", Y)




    path_ref = f"./merlDB/db/brdfs/{materiaux_res}.binary"
    path_test = f"./results/prediction/{materiaux_res}-pred.binary"

    ref = readMERLBRDF(path_ref)
    test = readMERLBRDF(path_test)

    # Helper to align axes of `arr` to match `target` (only permutes spatial axes)
    def align_to(target, arr):
        if target.shape == arr.shape:
            return arr
        t3 = target.shape[:3]
        a3 = arr.shape[:3]
        if sorted(t3) == sorted(a3):
            from itertools import permutations
            for p in permutations((0, 1, 2)):
                perm = tuple(list(p) + [3])
                permuted = np.transpose(arr, axes=perm)
                if permuted.shape[:3] == t3:
                    return permuted
        # fallback: try swapping first two axes
        try:
            permuted = np.transpose(arr, (1, 0, 2, 3))
            if permuted.shape == target.shape:
                return permuted
        except Exception:
            pass
        raise ValueError(f"Cannot align array of shape {arr.shape} to target shape {target.shape}")

    # Align test to reference if needed
    try:
        test_aligned = align_to(ref, test)
    except ValueError:
        # Try aligning reference to test instead
        try:
            ref_aligned = align_to(test, ref)
            ref, test = ref_aligned, test
            test_aligned = test
        except ValueError as e:
            raise

    # Determine MAX_I
    max_i = float(np.max(ref))

    # Calculate continuity scores
    print("\nCalculating continuous score for reference BRDF:")
    score_ref = score_continuite(torch.from_numpy(ref))
    print(f"Reference BRDF continuous score: {score_ref:.4f}")
    print("="*50)
    print("\nCalculating continuous score for test BRDF:")
    score_test = score_continuite(torch.from_numpy(test_aligned))
    print(f"Test BRDF continuous score: {score_test:.4f}")
    print("="*50)

    # Overall PSNR
    overall = calculate_psnr(ref, test_aligned, max_i=max_i)

    # Per-channel PSNR
    psnr_r = calculate_psnr(ref[..., 0], test_aligned[..., 0], max_i=max_i)
    psnr_g = calculate_psnr(ref[..., 1], test_aligned[..., 1], max_i=max_i)
    psnr_b = calculate_psnr(ref[..., 2], test_aligned[..., 2], max_i=max_i)

    # Grayscale PSNR
    ref_gray = np.mean(ref, axis=-1)
    test_gray = np.mean(test_aligned, axis=-1)
    psnr_gray = calculate_psnr(ref_gray, test_gray, max_i=max_i)

    # Overall SSIM
    ssim_overall = calculate_ssim(ref, test_aligned, data_range=max_i)
    
    # Per-channel SSIM
    ssim_r = calculate_ssim(ref[..., 0], test_aligned[..., 0], data_range=max_i)
    ssim_g = calculate_ssim(ref[..., 1], test_aligned[..., 1], data_range=max_i)
    ssim_b = calculate_ssim(ref[..., 2], test_aligned[..., 2], data_range=max_i)
    
    # Grayscale SSIM
    ssim_gray = calculate_ssim(ref_gray, test_gray, data_range=max_i)

    print(f"MAX_I = {max_i}")
    print()
    print(f"PSNR overall: {overall:.4f} dB")
    print(f"PSNR R:       {psnr_r:.4f} dB")
    print(f"PSNR G:       {psnr_g:.4f} dB")
    print(f"PSNR B:       {psnr_b:.4f} dB")
    print(f"PSNR Gray:    {psnr_gray:.4f} dB")
    print()
    print(f"SSIM overall: {ssim_overall:.4f}")
    print(f"SSIM R:       {ssim_r:.4f}")
    print(f"SSIM G:       {ssim_g:.4f}")
    print(f"SSIM B:       {ssim_b:.4f}")
    print(f"SSIM Gray:    {ssim_gray:.4f}")
    

    # Render if requested (single or multi-angle)
    if show_render or show_multi_angle:
        # Load medians for normalization
        median_dir_ref = "./merlDB/db/brdfs/"
        median_dir_test = "./results/prediction/"
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        renderer = Renderer(size=1000, save_path="results",nb_spheres=1,nb_tours=10,gamma=2.222)
        
        dbuilder_ref = db.DBuilder(interp_method="linear",db_path=median_dir_ref)
        dbuilder_test = db.DBuilder(interp_method="linear",db_path=median_dir_test)
        print(dbuilder_ref.list_db())

        mat_ref = materiaux_res
        mat_test = f"{materiaux_res}-pred"
        
        dbuilder_ref.load_mat(mat_ref)
        print("Loaded " + mat_ref)
        brdf = dbuilder_ref.brdf_function(mat_ref)

        dbuilder_test.load_mat(mat_test)
        print("Loaded " + mat_test)
        brdf_test = dbuilder_test.brdf_function(mat_test)
        
        # Show multi-angle view if requested
        if show_multi_angle:
            print("\n" + "="*50)
            print("Generating multi-angle sphere views...")
            print("="*50)
            
            # Define viewing angles: rotation around sphere
            angles = [
                (0, 0), (0, 2), (0, 4), (0, 6), (0, 8)
            ]
            angle_names = ["0°", "72°", "144°", "216°", "288°"]
            
            # Render spheres from multiple angles
            images_ref = render_sphere_multiple_angles(brdf, mat_ref, renderer, angles)
            images_test = render_sphere_multiple_angles(brdf_test, mat_test, renderer, angles)
            
            # Display comparison
            display_multi_angle_comparison(images_ref, images_test, materiaux_res, angle_names)
        
        # Show single render if requested
        if show_render:
            image_ref = renderer.sub_render(0,0,brdf,name=mat_ref,save=False,grey_levels=False)
            image_test = renderer.sub_render(0,0,brdf_test,name=mat_test,save=False,grey_levels=False)

            # afficher les images originales et reconstruites pour comparaison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(Image.fromarray((255*image_ref).astype(np.uint8)))
            axes[0].set_title(f"Original BRDF - {mat_ref}")
            axes[0].axis('off')
            axes[1].imshow(Image.fromarray((255*image_test).astype(np.uint8)))
            axes[1].set_title(f"Reconstructed BRDF - {mat_test}")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
    
    # Return PSNR and SSIM values
    # Return PSNR and SSIM values
    return {
        'material': materiaux_res,
        'psnr_overall': overall,
        'psnr_r': psnr_r,
        'psnr_g': psnr_g,
        'psnr_b': psnr_b,
        'psnr_gray': psnr_gray,
        'ssim_overall': ssim_overall,
        'ssim_r': ssim_r,
        'ssim_g': ssim_g,
        'ssim_b': ssim_b,
        'ssim_gray': ssim_gray,
        'max_i': max_i
    }


def test_all_materials():
    """Test PSNR on all MERL materials and compute average"""
    
    all_materials = [
        'alum-bronze', 'alumina-oxide', 'aluminium', 'aventurnine', 'beige-fabric', 
        'black-fabric', 'black-obsidian', 'black-oxidized-steel', 'black-phenolic', 
        'black-soft-plastic', 'blue-acrylic', 'blue-fabric', 'blue-metallic-paint', 
        'blue-metallic-paint2', 'blue-rubber', 'brass', 'cherry-235', 'chrome-steel', 
        'chrome', 'colonial-maple-223', 'color-changing-paint1', 'color-changing-paint2', 
        'color-changing-paint3', 'dark-blue-paint', 'dark-red-paint', 'dark-specular-fabric', 
        'delrin', 'fruitwood-241', 'gold-metallic-paint', 'gold-metallic-paint2', 
        'gold-metallic-paint3', 'gold-paint', 'gray-plastic', 'grease-covered-steel', 
        'green-acrylic', 'green-fabric', 'green-latex', 'green-metallic-paint', 
        'green-metallic-paint2', 'green-plastic', 'hematite', 'ipswich-pine-221', 
        'light-brown-fabric', 'light-red-paint', 'maroon-plastic', 'natural-209', 
        'neoprene-rubber', 'nickel', 'nylon', 'orange-paint', 'pearl-paint', 
        'pickled-oak-260', 'pink-fabric', 'pink-fabric2', 'pink-felt', 'pink-jasper', 
        'pink-plastic', 'polyethylene', 'polyurethane-foam', 'pure-rubber', 'purple-paint', 
        'pvc', 'red-fabric', 'red-fabric2', 'red-metallic-paint', 'red-phenolic', 
        'red-plastic', 'red-specular-plastic', 'silicon-nitrade', 'silver-metallic-paint', 
        'silver-metallic-paint2', 'silver-paint', 'special-walnut-224', 
        'specular-black-phenolic', 'specular-blue-phenolic', 'specular-green-phenolic', 
        'specular-maroon-phenolic', 'specular-orange-phenolic', 'specular-red-phenolic', 
        'specular-violet-phenolic', 'specular-white-phenolic', 'specular-yellow-phenolic', 
        'ss440', 'steel', 'teflon', 'tungsten-carbide', 'two-layer-gold', 'two-layer-silver', 
        'violet-acrylic', 'violet-rubber', 'white-acrylic', 'white-diffuse-bball', 
        'white-fabric', 'white-fabric2', 'white-marble', 'white-paint', 
        'yellow-matte-plastic', 'yellow-paint', 'yellow-phenolic', 'yellow-plastic'
    ]
    
    results = []
    failed_materials = []
    
    print(f"\n{'='*70}")
    print(f"Testing {len(all_materials)} materials...")
    print(f"{'='*70}\n")
    
    for i, material in enumerate(all_materials, 1):
        print(f"\n[{i}/{len(all_materials)}] Processing: {material}")
        print("-" * 70)
        
        try:
            result = psnr(materiaux_res=material, show_render=False)
            results.append(result)
            print(f"✓ Success - PSNR: {result['psnr_overall']:.4f} dB, SSIM: {result['ssim_overall']:.4f}")
        except Exception as e:
            print(f"✗ Failed - Error: {str(e)}")
            failed_materials.append(material)
    
    # Compute averages
    if results:
        print(f"\n\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        avg_overall = np.mean([r['psnr_overall'] for r in results])
        avg_r = np.mean([r['psnr_r'] for r in results])
        avg_g = np.mean([r['psnr_g'] for r in results])
        avg_b = np.mean([r['psnr_b'] for r in results])
        avg_gray = np.mean([r['psnr_gray'] for r in results])
        
        avg_ssim_overall = np.mean([r['ssim_overall'] for r in results])
        avg_ssim_r = np.mean([r['ssim_r'] for r in results])
        avg_ssim_g = np.mean([r['ssim_g'] for r in results])
        avg_ssim_b = np.mean([r['ssim_b'] for r in results])
        avg_ssim_gray = np.mean([r['ssim_gray'] for r in results])
        
        print(f"Successfully tested: {len(results)}/{len(all_materials)} materials")
        if failed_materials:
            print(f"Failed materials: {len(failed_materials)}")
            print(f"  {', '.join(failed_materials)}")
        
        print(f"\nAVERAGE PSNR VALUES:")
        print(f"  Overall: {avg_overall:.4f} dB")
        print(f"  R:       {avg_r:.4f} dB")
        print(f"  G:       {avg_g:.4f} dB")
        print(f"  B:       {avg_b:.4f} dB")
        print(f"  Gray:    {avg_gray:.4f} dB")
        
        print(f"\nAVERAGE SSIM VALUES:")
        print(f"  Overall: {avg_ssim_overall:.4f}")
        print(f"  R:       {avg_ssim_r:.4f}")
        print(f"  G:       {avg_ssim_g:.4f}")
        print(f"  B:       {avg_ssim_b:.4f}")
        print(f"  Gray:    {avg_ssim_gray:.4f}")
        
        # Save results to file
        results_file = "./results/psnr_all_materials.txt"
        with open(results_file, 'w') as f:
            f.write("PSNR and SSIM Results for All Materials\n")
            f.write("="*100 + "\n\n")
            f.write(f"Successfully tested: {len(results)}/{len(all_materials)} materials\n\n")
            
            for r in results:
                f.write(f"{r['material']:<30} ")
                f.write(f"PSNR Overall: {r['psnr_overall']:>7.4f} dB  ")
                f.write(f"R: {r['psnr_r']:>7.4f}  G: {r['psnr_g']:>7.4f}  ")
                f.write(f"B: {r['psnr_b']:>7.4f}  Gray: {r['psnr_gray']:>7.4f}  |  ")
                f.write(f"SSIM Overall: {r['ssim_overall']:>6.4f}  ")
                f.write(f"R: {r['ssim_r']:>6.4f}  G: {r['ssim_g']:>6.4f}  ")
                f.write(f"B: {r['ssim_b']:>6.4f}  Gray: {r['ssim_gray']:>6.4f}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write(f"\nAVERAGE PSNR VALUES:\n")
            f.write(f"  Overall: {avg_overall:.4f} dB\n")
            f.write(f"  R:       {avg_r:.4f} dB\n")
            f.write(f"  G:       {avg_g:.4f} dB\n")
            f.write(f"  B:       {avg_b:.4f} dB\n")
            f.write(f"  Gray:    {avg_gray:.4f} dB\n")
            
            f.write(f"\nAVERAGE SSIM VALUES:\n")
            f.write(f"  Overall: {avg_ssim_overall:.4f}\n")
            f.write(f"  R:       {avg_ssim_r:.4f}\n")
            f.write(f"  G:       {avg_ssim_g:.4f}\n")
            f.write(f"  B:       {avg_ssim_b:.4f}\n")
            f.write(f"  Gray:    {avg_ssim_gray:.4f}\n")
            
            if failed_materials:
                f.write(f"\nFailed materials ({len(failed_materials)}):\n")
                for mat in failed_materials:
                    f.write(f"  - {mat}\n")
        
        print(f"\nResults saved to: {results_file}")
        print(f"{'='*70}\n")
    
    return results, failed_materials


if __name__ == "__main__":
    # Test all materials
    #test_all_materials()
    
    # Test a single material with single render:
    #psnr(materiaux_res="pink-plastic", show_render=True)
    
    # Test with multi-angle view:
    #psnr(materiaux_res="alum-bronze", show_multi_angle=True)
    
    # Test with both single and multi-angle:
    psnr(materiaux_res="red-plastic", show_render=False, show_multi_angle=True)
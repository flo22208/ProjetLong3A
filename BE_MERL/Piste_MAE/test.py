# Importation des librairies

# Résolution du conflit OpenMP entre bibliothèques (PyTorch, NumPy, scikit-learn)
import os

from merlFunctions import readMERLBRDF
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.utils.data as torchdata
import torchsummary
import neural_data as nd
import merlDB.database as db
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import mae_model_brdf
import util
import pandas as pd
from PIL import Image

from merlDB.rendering import Renderer




def compress_binary(mat, output, x_grid, NoL, NoV, encoder, size):
    z = util.encode_material_partial(mat, x_grid, NoL, NoV, encoder, size)
    np.save(output, z)
    return z


def decompress_binary(output, z, x_grid, NoL, NoV, decoder):
    brdf = util.decode_material(x_grid, NoL, NoV, z, decoder, False)
    util.save_BRDF(output, brdf)
    return brdf


def psnr(a: np.ndarray, b: np.ndarray, max_i: float) -> float:
    diff = a - b
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_i * max_i) / mse)


def load():
    # Si c'est vide ça veut dire qu'on fait l'entraînement sur tous les matériaux, sinon remplacer par un matériaux
    materiaux= ""

    # Pourcentage de triplets (R,G,B) masqués (recommandé 0.5)
    ratio_masking = 0.5

    # Quantité de données à utiliser pour l'entraînement (min 1 max 100)
    train_size = 100

    # Batch size
    batch_size = 20

    # Epochs
    epochs = 200

    # poids ajouté (valeur par défaut = 1)
    poids = 2 

    # taille de l'espace latent (valeur par défaut = 768, il faut que ce soit divisible par le nombre de tête d'attention (par défaut 12))
    embed_dim = 768

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # Récupération du dataset


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='results/')
    parser.add_argument('--merldir', default='../Deepclustering_supervised/merlDB/db/brdfs/')
    parser.add_argument('--mediandir', default='../Deepclustering_supervised/merlDB/db/')
    parser.add_argument('--batch_size', default=batch_size)
    parser.add_argument('--train_size', default=train_size)
    parser.add_argument('--monomat', default=None)
    parser.add_argument('--epochs', default=epochs)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--load_previous', default=False)
    parser.add_argument('--materiau', default=materiaux) 
    args, unknown = parser.parse_known_args()


    medians = db.readbin(args.mediandir + 'merl_median.binary')

    def median_mapping(albedos, epsilon=0.002):
        return np.log((albedos + epsilon)/(medians + epsilon) + 1)
        
    dbuilder = db.DBuilder(db_path=args.merldir, albedo_mapping=median_mapping)
    #test_dbuilder = db.DBuilder(db_path=args.merldir, albedo_mapping=median_mapping)
    ldb = dbuilder.list_db()
    #print(ldb)
    if args.load_previous:
        print('LOADING MODEL')
        train_info = np.load(args.outdir + 'train_info.npy', allow_pickle=True).item()
        mats = train_info.get('train_mats')
    else:
        if args.materiau != "" :
            mats = [args.materiau]
            #test_mats = [args.materiau]
        else :
            mats_idx = np.random.choice(range(len(ldb)), int(args.train_size), replace=False)
            mats = [ldb[idx] for idx in mats_idx]
            #test_mats = [mat for mat in ldb if mat not in mats]

    if not args.monomat is None:
        print("problème")
        mats = [args.monomat]

    #print(mats)
    for mat in tqdm(mats, 'Loading mats'):
        if mat != "Rea" :
            dbuilder.load_mat(mat)

    """
    if torch.cuda.is_available():
        for mat in tqdm(test_mats, 'Loading test mats'):
            if mat != "Rea" :
                test_dbuilder.load_mat(mat)
                #dbuilder.load_mat(mat)
    """
    dataset = nd.DBridge(dbuilder, device)
    #test_dataset = nd.DBridge(test_dbuilder, device)
    del dbuilder
    #del test_dbuilder


    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae = mae_model_brdf.MaskedAutoencoderViT3D(poids=poids, embed_dim=embed_dim)
    mae.load_state_dict(torch.load("./results/mae.pt"))
    mae.to(device)
    mae.eval()

    # 2. Use lists instead of pre-allocated tensors
    # This avoids shape mismatch errors (e.g. 109 vs 433 tokens)
    list_latents = []
    list_names = []

    print("Extracting features...")
    # Using a DataLoader handles batching automatically
    loader = torchdata.DataLoader(dataset, batch_size=1, shuffle=False)

    for data in tqdm(loader):
        # Extract data
        X = data['values'].to(device)
        
        # DataLoader wraps strings in a tuple, extend flattens them into the list
        list_names.extend(data['mat'])

        with torch.no_grad():
            # Output shape: (Batch, Num_Tokens, 768)
            latent = mae.forward_features(X)
            
            # Move to CPU immediately to save GPU memory
            list_latents.append(latent.cpu())

    # 3. Concatenate and Save
    # Stack all latents into one big tensor: (Total_Samples, Num_Tokens, Embed_Dim)
    representations_latentes = torch.cat(list_latents, dim=0)

    print(f"Final Tensor Shape: {representations_latentes.shape}")

    # Save the tensor
    torch.save(representations_latentes, "./results/representations_latentes.pt")

    # Save the list of strings directly
    torch.save(list_names, "./results/noms_refs_representations_latentes.pt")

def visualize():
    # Paramètres du t-SNE
    perplexity_val = 10      
    n_iter_val = 5000        
    early_exag_val = 20      
    metric_val = 'cosine'

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D


    latents = torch.load("./results/representations_latentes.pt", map_location='cpu')

    names = torch.load("./results/noms_refs_representations_latentes.pt")


    assert len(latents) == len(names), f"Mismatch: {len(latents)} latents vs {len(names)} names"

    global_reps = latents[:, 0, :].numpy()

    print(f"Global representations shape: {global_reps.shape}")

    # === PCA 3D ===
    print("Calcul de l'ACP en 3D...")
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(global_reps)
    
    # Récupérer les pourcentages de variance expliquée
    var_explained = pca.explained_variance_ratio_ * 100
    
    # Visualisation PCA 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                        alpha=0.7, c='red', edgecolors='white', s=50)
    
    # Ajouter les labels des matériaux
    for i, name in enumerate(names):
        ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], name, fontsize=8)
    
    # Ajouter les pourcentages de variance sur les axes
    ax.set_xlabel(f'PC1 ({var_explained[0]:.2f}%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.2f}%)', fontsize=12, fontweight='bold')
    ax.set_zlabel(f'PC3 ({var_explained[2]:.2f}%)', fontsize=12, fontweight='bold')
    
    ax.set_title(f'ACP 3D de l\'espace latent ({len(names)} matériaux)\nVariance totale expliquée: {sum(var_explained):.2f}%', 
                fontsize=16)
    
    plt.tight_layout()
    plt.savefig("./results/pca_3d_visualization.png", dpi=300)
    plt.show()

    # === t-SNE 2D ===
    print("Calcul du t-SNE optimisé...")

    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity_val, 
        early_exaggeration=early_exag_val,
        max_iter=n_iter_val,
        metric=metric_val, 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )

    X_embedded = tsne.fit_transform(global_reps)


    plt.figure(figsize=(14, 8))
    ax = plt.gca()


    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, c='blue', edgecolors='white')


    plt.title(f'Résultat du t-SNE sur l\'espace latent ({len(names)} matériaux)', fontsize=16)

    texts = []
    for i, name in enumerate(names):
        txt = ax.annotate(name, (X_embedded[i, 0], X_embedded[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        texts.append(txt)


    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.savefig("./results/latent_space_visualization.png", dpi=300)
    plt.show()


def entrainement():
    # Si c'est vide ça veut dire qu'on fait l'entraînement sur tous les matériaux, sinon remplacer par un matériaux
    materiaux= ""

    # Pourcentage de triplets (R,G,B) masqués (recommandé 0.5)
    ratio_masking = 0.5

    # Quantité de données à utiliser pour l'entraînement (min 1 max 100)
    train_size = 100

    # Batch size
    batch_size = 8

    # Epochs
    epochs = 10

    # poids ajouté (valeur par défaut = 1)
    poids = 2 

    # taille de l'espace latent (valeur par défaut = 768, il faut que ce soit divisible par le nombre de tête d'attention (par défaut 12))
    dim_latent = 768

    # norme de la fontion de perte (par défaut 2, sinon mettre 1 pour une loss L1, autre chose va faire crash le programme)
    norme = 2

    def get_args_parser():
        parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
        parser.add_argument('--batch_size', default=64, type=int,
                            help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
        parser.add_argument('--epochs', default=400, type=int)
        parser.add_argument('--accum_iter', default=1, type=int,
                            help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

        # Model parameters
        parser.add_argument('--model', default='mae_brdf', type=str, metavar='MODEL',
                            help='Name of model to train')

        parser.add_argument('--input_size', default=(90, 90, 180), type=int,
                            help='images input size')

        parser.add_argument('--mask_ratio', default=ratio_masking, type=float,
                            help='Masking ratio (percentage of removed patches).')

        parser.add_argument('--norm_pix_loss', action='store_true',
                            help='Use (per-patch) normalized pixels as targets for computing loss')
        parser.set_defaults(norm_pix_loss=False)

        # Optimizer parameters
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help='weight decay (default: 0.05)')

        parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                            help='learning rate (absolute lr)')
        parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                            help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
        parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0')

        parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                            help='epochs to warmup LR')

        # Dataset parameters /!\ à modifier quand on aura le dataset
        parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                            help='dataset path')

        parser.add_argument('--output_dir', default='./output_dir',
                            help='path where to save, empty for no saving')
        parser.add_argument('--log_dir', default='./output_dir',
                            help='path where to tensorboard log')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--resume', default='',
                            help='resume from checkpoint')

        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--num_workers', default=10, type=int)
        parser.add_argument('--pin_mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
        parser.set_defaults(pin_mem=True)

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')

        return parser


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='results/')
    parser.add_argument('--merldir', default='../Deepclustering_supervised/merlDB/db/brdfs/')
    parser.add_argument('--mediandir', default='../Deepclustering_supervised/merlDB/db/')
    parser.add_argument('--batch_size', default=batch_size)
    parser.add_argument('--train_size', default=train_size)
    parser.add_argument('--monomat', default=None)
    parser.add_argument('--epochs', default=epochs)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--load_previous', default=False)
    parser.add_argument('--materiau', default=materiaux) 
    args, unknown = parser.parse_known_args()

    medians = db.readbin(args.mediandir + 'merl_median.binary')

    def median_mapping(albedos, epsilon=0.002):
        return np.log((albedos + epsilon)/(medians + epsilon) + 1)

    print(dim_latent)
    assert dim_latent % 2 == 0
    mae = mae_model_brdf.MaskedAutoencoderViT3D(poids=poids, embed_dim=dim_latent, norme=norme)

    dbuilder = db.DBuilder(db_path=args.merldir, albedo_mapping=median_mapping)
    test_dbuilder = db.DBuilder(db_path=args.merldir, albedo_mapping=median_mapping)
    ldb = dbuilder.list_db()
    print(ldb)
    if args.load_previous:
        print('LOADING MODEL')
        train_info = np.load(args.outdir + 'train_info.npy', allow_pickle=True).item()
        mats = train_info.get('train_mats')
    else:
        if args.materiau != "" :
            mats = [args.materiau]
            test_mats = [args.materiau]
        else :
            mats_idx = np.random.choice(range(len(ldb)), int(args.train_size), replace=False)
            mats = [ldb[idx] for idx in mats_idx]
            test_mats = [mat for mat in ldb if mat not in mats]

    if not args.monomat is None:
        print("problème")
        mats = [args.monomat]

    print(mats)
    for mat in tqdm(mats, 'Loading mats'):
        if mat != "Rea" :
            dbuilder.load_mat(mat)

    if torch.cuda.is_available():
        for mat in tqdm(test_mats, 'Loading test mats'):
            if mat != "Rea" :
                test_dbuilder.load_mat(mat)

    dataset = nd.DBridge(dbuilder, device)
    test_dataset = nd.DBridge(test_dbuilder, device)
    del dbuilder
    del test_dbuilder

    """
    if not torch.cuda.is_available():
        merl_ae = nd.MockMerlAE()
    else:
        merl_ae = nd.MerlAutoEncoder()
        if args.load_previous:
            merl_ae.load_state_dict(torch.load(args.outdir + 'melr_ae.pt'))
    """

    if args.load_previous:
            mae.load_state_dict(torch.load(args.outdir + 'mae.pt'))

    #trainer = nd.MerlAETrainer(device, dataset, mae=merl_ae, lr=float(args.lr), batch_size=int(args.batch_size), shuffle=True)
    args2 = get_args_parser()
    args2, unknown = args2.parse_known_args()
    trainer = nd.MAETrainer(device, dataset, mae=mae, lr=float(args.lr), batch_size=int(args.batch_size), shuffle=True, args=args2)

    torchsummary.summary(trainer.mae, nd.INPUT_SHAPE)


    for epoch in range(int(args.epochs)):
        print('EPOCH',epoch)
        trainer.train_epoch()
        total_preds = 0
        total_correct_preds = 0
        for i,data in enumerate(torchdata.DataLoader(test_dataset)):
            X = data['values']
            _, out, _ , latent = trainer.mae(X)
    
    print("saiving model...")
    def save_model():
        if torch.cuda.is_available() or True:
            if args.load_previous:
                train_info = np.load(args.outdir + 'train_info.npy', allow_pickle=True).item()
                train_info['losses'] = train_info['losses'] + trainer.losses
            else:
                train_info = {'losses' : trainer.losses, 'train_mats': mats}
            np.save(args.outdir + 'train_info.npy', train_info, allow_pickle=True)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            all_losses = train_info['losses']
            ax.plot(range(len(all_losses)), all_losses)
            fig.savefig(args.outdir + 'losses.png')
            plt.show()
            plt.close(fig)
            torch.save(trainer.mae.state_dict(), args.outdir + 'mae_2.pt')
    save_model()

def test_results():
    """
    Test the MAE model by reconstructing BRDFs and calculating PSNR
    """
    # Modèle paramètres
    poids = 2 
    dim_latent = 768
    norme = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Charger le modèle MAE
    mae = mae_model_brdf.MaskedAutoencoderViT3D(poids=poids, embed_dim=dim_latent, norme=norme)
    mae.load_state_dict(torch.load("./results/mae.pt"))
    mae.to(device)
    mae.eval()
    print("Model loaded successfully")
    
    # Paramètres de test
    materials_to_test = ["alum-bronze", "beige-fabric", "blue-acrylic", "gold-metallic-paint", "green-plastic", "pink-fabric"]
    mediandir = '../Deepclustering_supervised/merlDB/db/brdfs/'
    merldirbinary = '../Deepclustering_supervised/merlDB/db/brdfs/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--merldir', default='merlDB/db/brdfs/')
    parser.add_argument('--outdir', default='temp/frames/')
    parser.add_argument('--mat', default=None)
    parser.add_argument('--mediandir', default='../Deepclustering_supervised/merlDB/db/')
    args, unknown = parser.parse_known_args()

    renderer = Renderer(size=1000, save_path=args.outdir,nb_spheres=1,nb_tours=10,gamma=2.222)
    dbuilder = db.DBuilder(interp_method="linear",db_path=mediandir)
    print(dbuilder.list_db())

    # Load medians for normalization
    medians = db.readbin(args.mediandir + 'merl_median.binary')

    mat = "blue-acrylic"
    dbuilder.load_mat(mat)
    print("Loaded " + mat)
    brdf = dbuilder.brdf_function(mat)
    image = renderer.sub_render(0,0,brdf,name=mat,save=False,grey_levels=False)
    
    #plt.imshow(Image.fromarray((255*image).astype(np.uint8)))
    #plt.title(f"Original BRDF - {mat}")
    #plt.show()
    
    # Reconstruct the BRDF using the MAE model
    with torch.no_grad():
        # Get raw BRDF data (90, 90, 180, 3)
        brdf_raw = db.readbin(os.path.join(merldirbinary, mat + ".binary"))
        
        # Apply median mapping (same as during training)
        epsilon = 0.002
        brdf_norm = np.log((brdf_raw + epsilon)/(medians + epsilon) + 1)
        
        # Convert to tensor and permute to (B, C, D, H, W)
        brdf_tensor = torch.from_numpy(brdf_norm).float().to(device)
        brdf_tensor = brdf_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, 90, 90, 180)
        
        # Encode and decode (no masking for reconstruction)
        latent, _, ids_restore = mae.forward_encoder(brdf_tensor, mask_ratio=0)
        pred = mae.forward_decoder(latent, ids_restore)
        
        # Unpatchify to get back to volume shape
        reconstructed_tensor = mae.unpatchify(pred)  # (1, 3, 90, 90, 180)
        
        # Convert back to numpy and permute to (D, H, W, C)
        reconstructed_norm = reconstructed_tensor.squeeze(0).permute(3, 1, 2, 0).cpu().numpy()
        
        # Inverse median mapping to get back to BRDF values
        reconstructed_brdf_raw = (np.exp(reconstructed_norm) - 1) * (medians + epsilon) - epsilon
        
        # Clip negative values
        reconstructed_brdf_raw = np.clip(reconstructed_brdf_raw, 0, None)
        
        # Create BRDF function from reconstructed data
        # The renderer expects brdf(coords) where coords is (N, 3) with [phi, theta_d, theta_h]
        def reconstructed_brdf(coords):
            """
            coords: (N, 3) array with [phi, theta_d, theta_h]
            Returns: (N, 3) array of RGB values
            """
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)
            
            phi = coords[:, 0]
            theta_d = coords[:, 1]
            theta_h = coords[:, 2]
            
            # Map to BRDF indices
            idx_phi = (phi * 180 / np.pi).astype(int) % 180
            idx_theta_d = np.clip((theta_d * 90 / (np.pi/2)).astype(int), 0, 89)
            idx_theta_h = np.clip((theta_h * 90 / (np.pi/2)).astype(int), 0, 89)
            
            # Extract RGB values
            result = reconstructed_brdf_raw[idx_theta_h, idx_theta_d, idx_phi]
            return result

        # Render reconstructed BRDF
        image_reconstructed = renderer.sub_render(0, 0, reconstructed_brdf, name=f"{mat}_reconstructed", save=False, grey_levels=False)

    # Calculate PSNR between original and reconstructed images
    ## mettre les images en niveau de gris
    image_gray = np.mean(image, axis=-1)
    image_reconstructed_gray = np.mean(image_reconstructed, axis=-1)

    psnr_value = psnr(image_gray, image_reconstructed_gray, max_i=1.0)
    print(f"PSNR for {mat}: {psnr_value:.2f} dB")
    
    # afficher les images originales et reconstruites pour comparaison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_gray, cmap='gray')
    axes[0].set_title(f"Original BRDF - {mat}")
    axes[0].axis('off')
    axes[1].imshow(image_reconstructed_gray, cmap='gray')
    axes[1].set_title(f"Reconstructed BRDF - {mat}\nPSNR: {psnr_value:.2f} dB")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(args.outdir + 'comparison.png')
    plt.show()
    plt.close(fig)


def new_test_results():
    """
    Test the MAE model by reconstructing BRDFs and calculating PSNR
    """

    poids = 2 
    dim_latent = 768
    norme = 2
    materiaux_res = "blue-acrylic"
    merldir = "../Deepclustering_supervised/merlDB/db/brdfs/"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Charger le modèle MAE
    mae = mae_model_brdf.MaskedAutoencoderViT3D(poids=poids, embed_dim=dim_latent, norme=norme)
    mae.load_state_dict(torch.load("./results/mae.pt"))
    mae.to(device)
    mae.eval()
    medians = db.readbin('./merlDB/db/merl_median.binary')

    def median_mapping(albedos, epsilon=0.002):
        return np.log((albedos + epsilon)/(medians + epsilon) + 1)

    def median_unmapping(mapped_albedos, epsilon=0.002):
        return (np.exp(mapped_albedos) -1) * (medians + epsilon)  - epsilon
    
    from merlDB.database import writebin, readbin

    save_dbuilder = db.DBuilder(db_path=merldir, albedo_mapping=median_mapping)
    save_dbuilder.load_mat(materiaux_res)
    save_dataset = nd.DBridge(save_dbuilder, device)
    del save_dbuilder
    for i,data in enumerate(torchdata.DataLoader(save_dataset)):
            X = data['values']
            _, prediction, _ , latent = mae(X)
            prediction = mae.unpatchify(prediction)
            Y = torch.swapaxes(prediction.squeeze(), 0, -1).detach().cpu().numpy()
            Y = median_unmapping(Y)
            writebin(f"./results/prediction/{materiaux_res}-pred.binary", Y)




    path_ref = "../Deepclustering_supervised/merlDB/db/brdfs/blue-acrylic.binary"
    path_test = "results/prediction/blue-acrylic-pred.binary"

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

    # Overall PSNR
    overall = psnr(ref, test_aligned, max_i=max_i)

    # Per-channel PSNR
    psnr_r = psnr(ref[..., 0], test_aligned[..., 0], max_i=max_i)
    psnr_g = psnr(ref[..., 1], test_aligned[..., 1], max_i=max_i)
    psnr_b = psnr(ref[..., 2], test_aligned[..., 2], max_i=max_i)

    # Grayscale PSNR
    ref_gray = np.mean(ref, axis=-1)
    test_gray = np.mean(test_aligned, axis=-1)
    psnr_gray = psnr(ref_gray, test_gray, max_i=max_i)

    print(f"MAX_I = {max_i}")

    print(f"PSNR overall: {overall:.4f} dB")
    print(f"PSNR R:       {psnr_r:.4f} dB")
    print(f"PSNR G:       {psnr_g:.4f} dB")
    print(f"PSNR B:       {psnr_b:.4f} dB")
    print(f"PSNR Gray:    {psnr_gray:.4f} dB")
    

    # Render
    # Load medians for normalization
    median_dir_ref = "../Deepclustering_supervised/merlDB/db/brdfs/"
    median_dir_test = "results/prediction/"

    renderer = Renderer(size=1000, save_path="results",nb_spheres=1,nb_tours=10,gamma=2.222)
    
    dbuilder_ref = db.DBuilder(interp_method="linear",db_path=median_dir_ref)
    dbuilder_test = db.DBuilder(interp_method="linear",db_path=median_dir_test)
    print(dbuilder_ref.list_db())

    mat_ref = "blue-acrylic"
    mat_test = "blue-acrylic-pred"
    
    dbuilder_ref.load_mat(mat_ref)
    print("Loaded " + mat_ref)
    brdf = dbuilder_ref.brdf_function(mat_ref)
    image_ref = renderer.sub_render(0,0,brdf,name=mat_ref,save=False,grey_levels=False)

    dbuilder_test.load_mat(mat_test)
    print("Loaded " + mat_test)
    brdf_test = dbuilder_test.brdf_function(mat_test)
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



if __name__ == "__main__":
    #entrainement()
    #load()
    #visualize()
    new_test_results()
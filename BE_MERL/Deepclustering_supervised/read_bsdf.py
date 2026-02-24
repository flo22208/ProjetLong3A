"""
Script pour lire et afficher toutes les informations d'un fichier .bsdf (format Mitsuba tensor_file)
"""

import struct
import numpy as np
import sys
import os


# Mapping des codes de type de données
DTYPE_MAP = {
    1: ('uint8', np.uint8),
    2: ('int8', np.int8),
    3: ('uint16', np.uint16),
    4: ('int16', np.int16),
    5: ('uint32', np.uint32),
    6: ('int32', np.int32),
    7: ('uint64', np.uint64),
    8: ('int64', np.int64),
    9: ('float16', np.float16),
    10: ('float32', np.float32),
    11: ('float64', np.float64),
}


def read_bsdf(filename):
    """Lit un fichier .bsdf et retourne toutes ses informations
    
    Args:
        filename (str): Chemin vers le fichier .bsdf
        
    Returns:
        dict: Dictionnaire contenant toutes les informations du fichier
    """
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas")
    
    info = {
        'filename': filename,
        'filesize': os.path.getsize(filename),
        'magic': None,
        'version': None,
        'num_fields': 0,
        'fields': {}
    }
    
    with open(filename, 'rb') as f:
        # Lire le magic string (12 bytes)
        magic = f.read(12)
        info['magic'] = magic.decode('utf-8').rstrip('\x00')
        
        if info['magic'] != 'tensor_file':
            raise ValueError(f"Format invalide: magic = '{info['magic']}' (attendu 'tensor_file')")
        
        # Lire la version (2 bytes)
        version_major, version_minor = struct.unpack('<BB', f.read(2))
        info['version'] = f"{version_major}.{version_minor}"
        
        # Lire le nombre de champs
        num_fields = struct.unpack('<I', f.read(4))[0]
        info['num_fields'] = num_fields
        
        # Lire les headers des champs
        field_headers = []
        for i in range(num_fields):
            # Longueur du label
            label_len = struct.unpack('<H', f.read(2))[0]
            # Label
            label = f.read(label_len).decode('utf-8')
            # Nombre de dimensions
            ndim = struct.unpack('<H', f.read(2))[0]
            # Type de données
            dtype_code = struct.unpack('<B', f.read(1))[0]
            # Offset des données
            data_offset = struct.unpack('<Q', f.read(8))[0]
            # Shape (dimensions)
            shape = []
            for j in range(ndim):
                dim = struct.unpack('<Q', f.read(8))[0]
                shape.append(dim)
            
            dtype_name, dtype_np = DTYPE_MAP.get(dtype_code, (f'unknown_{dtype_code}', None))
            
            field_headers.append({
                'label': label,
                'ndim': ndim,
                'dtype_code': dtype_code,
                'dtype_name': dtype_name,
                'dtype_np': dtype_np,
                'data_offset': data_offset,
                'shape': tuple(shape),
                'size': int(np.prod(shape)) if shape else 0
            })
        
        # Lire les données de chaque champ
        for header in field_headers:
            label = header['label']
            dtype_np = header['dtype_np']
            shape = header['shape']
            data_offset = header['data_offset']
            size = header['size']
            
            if dtype_np is None:
                info['fields'][label] = {
                    'header': header,
                    'data': None,
                    'error': f"Type de données inconnu: {header['dtype_code']}"
                }
                continue
            
            # Se positionner à l'offset des données
            f.seek(data_offset)
            
            # Lire les données
            try:
                data = np.fromfile(f, dtype=dtype_np, count=size)
                if len(shape) > 0 and size > 0:
                    data = data.reshape(shape)
                
                field_info = {
                    'header': header,
                    'data': data,
                }
                
                # Ajouter des statistiques pour les données numériques
                if dtype_np in [np.float16, np.float32, np.float64]:
                    valid_data = data[np.isfinite(data)]
                    if len(valid_data) > 0:
                        field_info['stats'] = {
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'median': float(np.median(valid_data)),
                            'nan_count': int(np.sum(np.isnan(data))),
                            'inf_count': int(np.sum(np.isinf(data))),
                            'valid_ratio': len(valid_data) / data.size
                        }
                elif dtype_np in [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32]:
                    field_info['stats'] = {
                        'min': int(np.min(data)),
                        'max': int(np.max(data)),
                        'mean': float(np.mean(data)),
                        'unique_values': len(np.unique(data))
                    }
                
                info['fields'][label] = field_info
                
            except Exception as e:
                info['fields'][label] = {
                    'header': header,
                    'data': None,
                    'error': str(e)
                }
    
    return info


def print_bsdf_info(info, verbose=False):
    """Affiche les informations d'un fichier .bsdf de manière formatée
    
    Args:
        info (dict): Informations retournées par read_bsdf()
        verbose (bool): Si True, affiche les données complètes
    """
    
    print("=" * 80)
    print(f"INFORMATIONS DU FICHIER BSDF: {os.path.basename(info['filename'])}")
    print("=" * 80)
    print(f"\nFichier: {info['filename']}")
    print(f"Taille: {info['filesize']:,} bytes ({info['filesize']/1024:.2f} KB)")
    print(f"Magic: '{info['magic']}'")
    print(f"Version: {info['version']}")
    print(f"Nombre de champs: {info['num_fields']}")
    
    print("\n" + "=" * 80)
    print("CHAMPS")
    print("=" * 80)
    
    for i, (label, field) in enumerate(info['fields'].items(), 1):
        header = field['header']
        print(f"\n[{i}] Champ: {label}")
        print(f"    Type: {header['dtype_name']} (code {header['dtype_code']})")
        print(f"    Dimensions: {header['ndim']}")
        print(f"    Shape: {header['shape']}")
        print(f"    Taille: {header['size']:,} éléments")
        print(f"    Offset données: {header['data_offset']:,} bytes")
        
        if 'error' in field:
            print(f"    ⚠ ERREUR: {field['error']}")
            continue
        
        if field['data'] is not None:
            data = field['data']
            print(f"    Shape effective: {data.shape}")
            
            # Afficher les statistiques
            if 'stats' in field:
                stats = field['stats']
                print(f"    Statistiques:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"      - {key}: {value:.6f}")
                    else:
                        print(f"      - {key}: {value}")
            
            # Afficher les données si verbose
            if verbose:
                print(f"    Données:")
                if header['dtype_name'] == 'uint8' and label == 'description':
                    # Cas spécial pour la description (texte)
                    try:
                        text = data.tobytes().decode('utf-8', errors='ignore')
                        print(f"      '{text}'")
                    except:
                        print(f"      {data[:100]}..." if len(data) > 100 else f"      {data}")
                else:
                    # Afficher un échantillon des données
                    if data.size <= 20:
                        print(f"      {data}")
                    else:
                        flat = data.flatten()
                        print(f"      Premiers éléments: {flat[:10]}")
                        print(f"      Derniers éléments: {flat[-10:]}")
    
    print("\n" + "=" * 80)


def main():
    """Fonction principale"""
    
    if len(sys.argv) < 2:
        print("Usage: python read_bsdf.py <fichier.bsdf> [--verbose]")
        print("\nExemple:")
        print("  python read_bsdf.py matpreview/temp_brdf.bsdf")
        print("  python read_bsdf.py matpreview/cc_blue_agat_rgb.bsdf --verbose")
        
        # Liste les fichiers .bsdf disponibles
        print("\nFichiers .bsdf disponibles:")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.bsdf'):
                    path = os.path.join(root, file)
                    size = os.path.getsize(path)
                    print(f"  - {path} ({size:,} bytes)")
        
        sys.exit(1)
    
    filename = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    try:
        info = read_bsdf(filename)
        print_bsdf_info(info, verbose=verbose)
        
        # Option pour sauvegarder les données dans un fichier numpy
        if '--save-numpy' in sys.argv:
            output_file = filename.replace('.bsdf', '_data.npz')
            data_dict = {label: field['data'] 
                        for label, field in info['fields'].items() 
                        if field['data'] is not None}
            np.savez(output_file, **data_dict)
            print(f"\n✓ Données sauvegardées dans: {output_file}")
        
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

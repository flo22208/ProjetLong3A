#Read BRDF
import numpy as np
import os.path as path

def readUTIABRDF(filename):
    """Reads a UTIA-type .bin file, containing a densely sampled BRDF
    
    Returns a 5-dimensional array (phi_v, theta_v, phi_i, theta_i, channel)"""

    print("Loading UTIA-BRDF: ",filename)
    try: 
        f = open(filename, "rb")
        vals = np.fromfile(f,np.float64,-1)
        f.close()
    except IOError:
        print("Cannot read file:", path.basename(filename))
        return
        
    BRDFVals = np.reshape(vals,(48, 6, 48, 6, 3),'F')
    BRDFVals[BRDFVals<0] = 0
    
    return BRDFVals

def readMERLBRDF(filename):
    """Reads a MERL-type .binary file, containing a densely sampled BRDF
    
    Returns a 4-dimensional array (phi_d, theta_h,theta_d channel)"""
    print("Loading MERL-BRDF: ",filename)
    try: 
        f = open(filename, "rb")
        dims = np.fromfile(f,np.int32,3)
        vals = np.fromfile(f,np.float64,-1)
        f.close()
    except IOError:
        print("Cannot read file:", path.basename(filename))
        return
        
    BRDFVals = np.swapaxes(np.reshape(vals,(dims[2], dims[1], dims[0], 3),'F'),1,2)
    BRDFVals *= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    BRDFVals[BRDFVals<0] = 0
    return BRDFVals


def readMERLBRDF_v2(filename):
    """Reads a MERL-type .binary file, containing a densely sampled BRDF
    
    Returns a 4-dimensional array (phi_d, theta_h,theta_d channel)"""
    try: 
        f = open(filename, "rb")
        dims = np.fromfile(f,np.int32,3)
        vals = np.fromfile(f,np.float64,-1)
        f.close()
    except IOError:
        print("Cannot read file:", path.basename(filename))
        return
        
    BRDFVals = np.swapaxes(np.reshape(vals,(dims[2], dims[1], dims[0], 3),'F'),1,2)
    BRDFVals *= (1/1500,1/1500,1/1500) #Colorscaling
    BRDFVals[BRDFVals<0] = 0
    return BRDFVals


def saveMERLBRDF(filename,BRDFVals,shape=(180,90,90),toneMap=True):
    "Saves a BRDF to a MERL-type .binary file"
    print("Saving MERL-BRDF: ",filename)
    BRDFVals = np.array(BRDFVals)   #Make a copy
    if(BRDFVals.shape != (np.prod(shape),3) and BRDFVals.shape != shape+(3,)):
        print("Shape of BRDFVals incorrect")
        return
        
    #Do MERL tonemapping if needed
    if(toneMap):
        BRDFVals /= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    
    #Are the values not mapped in a cube?
    if(BRDFVals.shape[1] == 3):
        BRDFVals = np.reshape(BRDFVals,shape+(3,))
        
    #Vectorize:
    vec = np.reshape(np.swapaxes(BRDFVals,1,2),(-1),'F')
    shape = [shape[2],shape[1],shape[0]]
    
    try: 
        f = open(filename, "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", path.basename(filename))
        return
        
    

def saveUTIABRDF(filename,BRDFVals,shape=(48, 6, 48, 6, 3)):
    "Saves a BRDF to an UTIA-type .binary file"
    print("Saving UTIA-BRDF: ",filename)
    BRDFVals = np.array(BRDFVals)   #Make a copy
    
    #Vectorize:
    vec = np.reshape(BRDFVals.reshape(shape),(-1),'F')
    
    try: 
        f = open(filename, "wb")
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", path.basename(filename))
        return


def saveBSDF(filename, BRDFVals):
    """Saves a BRDF tensor to Mitsuba tensor_file format
    
    BRDFVals shape after transpose in visualizer: (theta_h, theta_d, phi_d, 3)
    """
    import struct
    
    print("Saving BSDF for Mitsuba: ", filename)
    
    BRDFVals = np.array(BRDFVals, dtype=np.float32)
    BRDFVals = np.ascontiguousarray(BRDFVals)
    
    # Shape: (theta_h, theta_d, phi_d, 3)
    if len(BRDFVals.shape) != 4 or BRDFVals.shape[3] != 3:
        print("Error: BRDFVals must have 4 dimensions with 3 RGB channels")
        return
    
    # Analyze original data
    valid_mask = np.isfinite(BRDFVals)
    print(f"  Data validity: {np.sum(valid_mask) / BRDFVals.size * 100:.1f}% valid values")
    
    # Replace all invalid values with a small constant
    BRDFVals = np.nan_to_num(BRDFVals, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Normalize to [0.01, 0.5] range to avoid extreme values
    valid_vals = BRDFVals[BRDFVals > 0]
    if len(valid_vals) > 0:
        vmin, vmax = np.min(valid_vals), np.max(valid_vals)
        print(f"  Original valid range: [{vmin:.6f}, {vmax:.6f}]")
        
        # Remplace les valeurs négatives/nulles par une petite valeur epsilon
        BRDFVals = np.where(BRDFVals <= 0, 1e-6, BRDFVals)
        
        # Passage en log-space pour préserver la dynamique (BRDFs varient sur plusieurs ordres de grandeur)
        log_vals = np.log1p(BRDFVals)
        log_min = np.log1p(vmin)
        log_max = np.log1p(vmax)
        
        if log_max > log_min:
            # Remapping log-space vers [0.001, 1.0] — plage plus physique pour Mitsuba
            BRDFVals =  (log_vals - log_min) / (log_max - log_min)
        
        print(f"  Normalized range (log-space): min={BRDFVals.min():.6f}, max={BRDFVals.max():.6f}")

    # Clip léger juste pour la sécurité numérique, sans écraser la dynamique
    BRDFVals = np.clip(BRDFVals, 1e-6, 1.0)
    print(f"  Normalized range: min={np.min(BRDFVals):.6f}, max={np.max(BRDFVals):.6f}")
    
    ntheta_h, ntheta_d, nphi_d, _ = BRDFVals.shape
    print(f"  BRDF dimensions: theta_h={ntheta_h}, theta_d={ntheta_d}, phi_d={nphi_d}")
    
    try:
        with open(filename, "wb") as f:
            # Write tensor_file header
            f.write(b'tensor_file\0')  # Magic string (12 bytes)
            f.write(struct.pack('<BB', 1, 0))  # Version 1.0
            
            # Write number of fields (10 fields for powitacq)
            num_fields = 10
            f.write(struct.pack('<I', num_fields))
            
            # Keep track of field offsets
            field_offsets = {}
            
            # For isotropic: phi_i needs at least 2 entries
            nphi_i = 2
            ntheta_i = ntheta_d
            
            # Create sampling angles
            theta_i_data = np.linspace(0, np.pi/2, ntheta_i, dtype=np.float32)
            phi_i_data = np.array([0.0, np.pi], dtype=np.float32)  # Two entries for isotropic
            
            print(f"  Sampling: theta_i size={ntheta_i}, phi_i size={nphi_i}")
            
            # Use BRDF to create proper BSDF fields
            rgb_data_expanded = np.zeros((nphi_i, ntheta_i, 3, ntheta_h, ntheta_h), dtype=np.float32)

            for phi_i_idx in range(nphi_i):
                for theta_d_idx in range(ntheta_d):

                    # BRDF slice: (theta_h, phi_d, 3)
                    brdf_slice = BRDFVals[:, theta_d_idx, :, :]

                    # Option isotrope : moyenne seulement sur phi_d
                    brdf_theta_h_rgb = np.mean(brdf_slice, axis=1)  # (theta_h, 3)
                   
                    for c in range(3):
                        rgb_data_expanded[phi_i_idx, theta_d_idx, c, :, :] = \
                            brdf_theta_h_rgb[:, c][:, None]
        
            
                        
            # Create VNDF - use BRDF magnitude as proxy
            vndf_data = np.zeros((nphi_i, ntheta_i, ntheta_h, ntheta_h), dtype=np.float32)
            for phi_i_idx in range(nphi_i):
                for theta_d_idx in range(ntheta_d):
                    brdf_avg = np.mean(BRDFVals[:, theta_d_idx, :, :])  # Single scalar average
                    brdf_avg = np.clip(brdf_avg, 0.01, 1)
                    vndf_data[phi_i_idx, theta_d_idx, :, :] = brdf_avg
            
            # Create luminance - same as VNDF
            luminance_data = np.zeros((nphi_i, ntheta_i, ntheta_h, ntheta_h), dtype=np.float32)
            for phi_i_idx in range(nphi_i):
                for theta_d_idx in range(ntheta_d):
                    brdf_avg = np.mean(BRDFVals[:, theta_d_idx, :, :])  # Single scalar average
                    brdf_avg = np.clip(brdf_avg, 0.01, 1)
                    luminance_data[phi_i_idx, theta_d_idx, :, :] = brdf_avg
            
            # Create NDF and Sigma
            ndf_data = np.ones((nphi_i, ntheta_i), dtype=np.float32)
            sigma_data = np.ones((nphi_i, ntheta_i), dtype=np.float32) * 0.1
            
            # Description and metadata
            description = b"MERL Material - Real BRDF Data"
            valid_data = np.ones((nphi_i, ntheta_i, ntheta_h, ntheta_h), dtype=np.uint8)
            jacobian_data = np.array([1], dtype=np.uint8)
            
            fields = {
                'theta_i': theta_i_data,
                'phi_i': phi_i_data,
                'ndf': ndf_data,
                'sigma': sigma_data,
                'vndf': vndf_data,
                'rgb': rgb_data_expanded,
                'luminance': luminance_data,
                'description': np.frombuffer(description, dtype=np.uint8),
                'valid': valid_data,
                'jacobian': jacobian_data
            }
            
            # Final validation - ensure no NaN/Inf in any field
            print("  Validating all fields...")
            for field_name, field_data in fields.items():
                if field_data.dtype in (np.float32, np.float64):
                    nan_mask = np.isnan(field_data)
                    if np.any(nan_mask):
                        print(f"    WARNING: {field_name} has {np.sum(nan_mask)} NaN values, replacing...")
                        field_data = np.nan_to_num(field_data, nan=0.1)
                        fields[field_name] = field_data
                    inf_mask = np.isinf(field_data)
                    if np.any(inf_mask):
                        print(f"    WARNING: {field_name} has {np.sum(inf_mask)} Inf values, clipping...")
                        field_data = np.clip(field_data, -1.0, 1.0)
                        fields[field_name] = field_data
                    print(f"    {field_name}: shape={field_data.shape}, range=[{np.min(field_data):.4f}, {np.max(field_data):.4f}]")
            
            # Write field headers
            field_offsets = {}
            for field_name in ['theta_i', 'phi_i', 'ndf', 'sigma', 'vndf', 'rgb', 'luminance', 'description', 'valid', 'jacobian']:
                field_data = fields[field_name]
                label = field_name.encode('utf8')
                f.write(struct.pack('<H', len(label)))
                f.write(label)
                f.write(struct.pack('<H', len(field_data.shape)))  # ndim
                
                # Determine dtype code: 1=uint8, 10=float32
                if field_data.dtype == np.uint8:
                    dtype_code = 1
                elif field_data.dtype == np.float32:
                    dtype_code = 10
                else:
                    dtype_code = 10  # default to float32
                
                f.write(struct.pack('<B', dtype_code))
                print(f"    Writing {field_name}: dtype={field_data.dtype}, code={dtype_code}")
                
                # Store offset position - this is where we'll write the actual data offset later
                field_offsets[field_name] = f.tell()
                f.write(struct.pack('<Q', 0))  # placeholder for data offset
                
                # Write shape
                for dim in field_data.shape:
                    f.write(struct.pack('<Q', dim))
            
            # Write field data at aligned positions
            for field_name in ['theta_i', 'phi_i', 'ndf', 'sigma', 'vndf', 'rgb', 'luminance', 'description', 'valid', 'jacobian']:
                field_data = fields[field_name]
                
                # Align to 8-byte boundary
                current_pos = f.tell()
                aligned_pos = (current_pos + 7) // 8 * 8
                
                if aligned_pos > current_pos:
                    f.write(b'\0' * (aligned_pos - current_pos))
                
                # Update offset in header with the actual data position
                f.seek(field_offsets[field_name])
                f.write(struct.pack('<Q', aligned_pos))
                
                # Write data at aligned position
                f.seek(aligned_pos)
                field_data.astype(field_data.dtype).tofile(f)
            
        print(f"✓ BSDF saved successfully: {filename}")
        
    except Exception as e:
        print(f"✗ Error saving BSDF: {e}")
        import traceback
        traceback.print_exc()
        return
        
import numpy as np
from tqdm import tqdm
import merlDB.database as db
from numpyBRDF import BRDF, rusinkiewicz_to_LV
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

file = "results/merl_on_jax/jax.npz"

# read file
data = np.load(file, allow_pickle=True)

# Precompute angles (shared data)
RES_THETA_H = 90
RES_THETA_D = 90
RES_PHI_D = 180
MAX_THETA_H = 90
MAX_THETA_D = 90
MAX_PHI_D = 180

theta_hs = np.deg2rad(np.linspace(0, RES_THETA_H, MAX_THETA_H))
theta_ds = np.deg2rad(np.linspace(0, RES_THETA_D, MAX_THETA_D))
phi_ds = np.deg2rad(np.linspace(0, RES_PHI_D, MAX_PHI_D))

angles = np.zeros((MAX_THETA_H*MAX_THETA_D*MAX_PHI_D, 3))
idx = 0
for hi in range(MAX_THETA_H):
    for di in range(MAX_THETA_D):
        for pi in range(MAX_PHI_D):
            angles[idx] = [phi_ds[pi], theta_ds[di], theta_hs[hi]]
            idx += 1

# Global database builder (shared across processes)
dbuilder = db.DBuilder(interp_method="linear", db_path='merlDB/db/brdfs/')

def process_material(args):
    """Process one material: load BRDF, compute loss"""
    i, data_slice = args
    mat = data_slice[0]
    params = data_slice[1]
    
    dbuilder.load_mat(mat)
    brdf_function = dbuilder.brdf_function(mat)
    
    # Ground truth BRDF
    gt_brdf = brdf_function(angles)
    gt_brdf = gt_brdf / (1.0 + gt_brdf)  # tonemapping
    gt_brdf = gt_brdf.reshape((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3))
    
    # Disney BRDF
    brdf = np.zeros((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3), dtype=np.float32)
    for hi in range(MAX_THETA_H):
        for di in range(MAX_THETA_D):
            for pi in range(MAX_PHI_D):
                theta_h = theta_hs[hi]
                theta_d = theta_ds[di]
                phi_d = phi_ds[pi]
                
                L, V, N_vec, X, Y = rusinkiewicz_to_LV(theta_h, 0, theta_d, phi_d)
                vals = BRDF(
                    L, V, N_vec, X, Y,
                    params["baseColor"], params["metallic"], params["subsurface"],
                    params["specular"], params["roughness"], params["specularTint"],
                    params["anisotropic"], params["sheen"], params["sheenTint"],
                    params["clearcoat"], params["clearcoatGloss"]
                )
                if vals[0] != -1:
                    brdf[hi, di, pi] = vals
    
    brdf = brdf / (1.0 + brdf)  # tonemapping

    # MSE loss
    loss = np.mean((gt_brdf - brdf)**2)
    return mat, loss

if __name__ == "__main__":
    # Prepare material tasks
    num_materials = len(data["params"]) // 2
    material_tasks = [(i, data["params"][2*i:2*i+2]) for i in range(num_materials)]
    
    # Parallel processing
    all_losses = {}
    max_workers = mp.cpu_count()  # Use all CPU cores
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_mat = {
            executor.submit(process_material, task): task[1][0] 
            for task in material_tasks
        }
        
        # Progress bar for completed tasks
        with tqdm(total=len(material_tasks), desc="Processing materials") as pbar:
            for future in as_completed(future_to_mat):
                mat, loss = future.result()
                all_losses[mat] = loss
                print(f"{mat}: {loss}")
                pbar.update(1)
    
    print(all_losses)
    print("Average loss: ", np.mean(list(all_losses.values())))

encoder = {'blue-rubber': np.float64(0.007613820265115672), 'black-fabric': np.float64(0.005205645863110805), 'aventurnine': np.float64(0.006033124830322342), 'aluminium': np.float64(0.006920528570243747), 'blue-fabric': np.float64(0.004392195186793183), 'blue-metallic-paint2': np.float64(0.0056137193153397045), 'black-soft-plastic': np.float64(0.008039598551370519), 'brass': np.float64(0.006801605106341193), 'alum-bronze': np.float64(0.0048479442478769645), 'blue-metallic-paint': np.float64(0.008749966943254972), 'black-oxidized-steel': np.float64(0.006204923472838465), 'black-phenolic': np.float64(0.005612977680756819), 'black-obsidian': np.float64(0.006585105842612025), 'beige-fabric': np.float64(0.005433181629017319), 'alumina-oxide': np.float64(0.009558579634890818), 'blue-acrylic': np.float64(0.0050111226691843935), 'chrome-steel': np.float64(0.010887961402573408), 'dark-specular-fabric': np.float64(0.007857585814982247), 'colonial-maple-223': np.float64(0.007143569293567756), 'cherry-235': np.float64(0.005372957949975485), 'gold-metallic-paint': np.float64(0.009419800925003718), 'dark-blue-paint': np.float64(0.008627306234208175), 'chrome': np.float64(0.01134357507047289), 'color-changing-paint2': np.float64(0.006637661437528305), 'dark-red-paint': np.float64(0.011514494472983874), 'color-changing-paint1': np.float64(0.0061215506485907565), 'gold-paint': np.float64(0.007531806729022733), 'gold-metallic-paint2': np.float64(0.008119444181923863), 'color-changing-paint3': np.float64(0.006472842643872242), 'delrin': np.float64(0.008747302697733791), 'fruitwood-241': np.float64(0.004710099199002574), 'gold-metallic-paint3': np.float64(0.007299298811012535), 'gray-plastic': np.float64(0.005441179449656879), 'grease-covered-steel': np.float64(0.00810178962752851), 'green-latex': np.float64(0.002817200647465746), 'green-metallic-paint': np.float64(0.005297077732569194), 'green-acrylic': np.float64(0.006539303329656895), 'green-metallic-paint2': np.float64(0.0033200841371942625), 'green-fabric': np.float64(0.004479673668838867), 'hematite': np.float64(0.00403482259293492), 'light-red-paint': np.float64(0.01149481252686222), 'ipswich-pine-221': np.float64(0.0059833865091395485), 'green-plastic': np.float64(0.006157868929284892), 'light-brown-fabric': np.float64(0.006998593936031802), 'natural-209': np.float64(0.005621059603145085), 'maroon-plastic': np.float64(0.009632560698672308), 'nickel': np.float64(0.011328319876923891), 'neoprene-rubber': np.float64(0.00745307265051132), 'orange-paint': np.float64(0.01306504529270556), 'nylon': np.float64(0.009605562139508196), 'pearl-paint': np.float64(0.008517705307650182), 'pink-jasper': np.float64(0.006058552290015609), 'pink-fabric2': np.float64(0.006687780246049184), 'pickled-oak-260': np.float64(0.006849673160316164), 'pink-plastic': np.float64(0.014572994673363131), 'polyethylene': np.float64(0.013891113508853924), 'pink-felt': np.float64(0.009984327748229254), 'pink-fabric': np.float64(0.00792538088356109), 'pure-rubber': np.float64(0.009764569527851744), 'polyurethane-foam': np.float64(0.0030585631309081496), 'pvc': np.float64(0.005095171347029395), 'purple-paint': np.float64(0.005807018915384575), 'red-fabric2': np.float64(0.00849639618160197), 'red-fabric': np.float64(0.0053496856400878185), 'red-phenolic': np.float64(0.006711722282979202), 'red-metallic-paint': np.float64(0.004499302622841365), 'silicon-nitrade': np.float64(0.005441498672128506), 'silver-metallic-paint': np.float64(0.015097465256264792), 'red-specular-plastic': np.float64(0.010175881817917037), 'red-plastic': np.float64(0.008477450958495992), 'silver-paint': np.float64(0.007886255884954715), 'specular-green-phenolic': np.float64(0.006223054060861649), 'special-walnut-224': np.float64(0.007198466435443596), 'specular-blue-phenolic': np.float64(0.006241482067593433), 'silver-metallic-paint2': np.float64(0.013830120326502902), 'specular-maroon-phenolic': np.float64(0.009734554884696206), 'specular-black-phenolic': np.float64(0.006630998968580813), 'specular-orange-phenolic': np.float64(0.014314629458703902), 'specular-red-phenolic': np.float64(0.01395166365882152), 'specular-violet-phenolic': np.float64(0.007118662269329386), 'specular-white-phenolic': np.float64(0.010381657328675901), 'steel': np.float64(0.010326716616254887), 'tungsten-carbide': np.float64(0.009721140383091046), 'specular-yellow-phenolic': np.float64(0.014919234748496255), 'ss440': np.float64(0.008896930578792749), 'two-layer-gold': np.float64(0.007748137143709527), 'teflon': np.float64(0.00950173511608831), 'violet-acrylic': np.float64(0.007410948148295944), 'violet-rubber': np.float64(0.008779164865072917), 'two-layer-silver': np.float64(0.008932154371373485), 'white-diffuse-bball': np.float64(0.008489502152248142), 'white-fabric': np.float64(0.005367260358904829), 'white-fabric2': np.float64(0.0047536514994589475), 'white-acrylic': np.float64(0.011409750644539862), 'white-marble': np.float64(0.0077876838194150715), 'white-paint': np.float64(0.011517943059339153), 'yellow-matte-plastic': np.float64(0.008164392289119402), 'yellow-paint': np.float64(0.01235247533665893), 'yellow-phenolic': np.float64(0.009063115372300257), 'yellow-plastic': np.float64(0.012283196605913448)}
jax = {'colonial-maple-223': np.float64(0.0036388213686752713), 'grease-covered-steel': np.float64(0.003981499229904261), 'yellow-matte-plastic': np.float64(0.0072378827981598465), 'steel': np.float64(0.017207747517080447), 'green-metallic-paint2': np.float64(0.015156469729402378), 'specular-white-phenolic': np.float64(0.012723885925137809), 'pink-felt': np.float64(0.002980571540649801), 'light-brown-fabric': np.float64(0.000510878881306298), 'polyurethane-foam': np.float64(0.0007037290224280381), 'two-layer-gold': np.float64(0.0077135987560124405), 'specular-maroon-phenolic': np.float64(0.013735552677439308), 'red-metallic-paint': np.float64(0.002897042841350206), 'red-fabric2': np.float64(0.0009228004846158164), 'violet-rubber': np.float64(0.0025908561637202347), 'dark-specular-fabric': np.float64(0.002713992239749867), 'black-fabric': np.float64(0.00034219632514626876), 'chrome-steel': np.float64(0.017663327985495254), 'aventurnine': np.float64(0.012317029125058408), 'yellow-plastic': np.float64(0.0035908856561908315), 'alumina-oxide': np.float64(0.012108920823585776), 'gold-metallic-paint': np.float64(0.005628891825564004), 'pvc': np.float64(0.007081472763211906), 'white-paint': np.float64(0.007729474068177989), 'gray-plastic': np.float64(0.010076553285115919), 'yellow-phenolic': np.float64(0.012147111896890753), 'light-red-paint': np.float64(0.0030693719772180317), 'dark-blue-paint': np.float64(0.0019391604940481176), 'nickel': np.float64(0.003699313768308258), 'pure-rubber': np.float64(0.0034107793963928912), 'ss440': np.float64(0.01795076116703434), 'pink-fabric2': np.float64(0.0014328184112175327), 'yellow-paint': np.float64(0.002921498817642908), 'violet-acrylic': np.float64(0.013042850188613016), 'aluminium': np.float64(0.02055520442309228), 'white-fabric': np.float64(0.001290319815515968), 'brass': np.float64(0.017181122240095777), 'beige-fabric': np.float64(0.0011920851366591003), 'black-phenolic': np.float64(0.012771729168121372), 'specular-green-phenolic': np.float64(0.01392358871018823), 'silver-metallic-paint': np.float64(0.0063590416300414844), 'color-changing-paint3': np.float64(0.010936972560248432), 'pink-jasper': np.float64(0.012017327303291523), 'black-soft-plastic': np.float64(0.001300110268922204), 'hematite': np.float64(0.01501184009997238), 'teflon': np.float64(0.0045175058162387), 'white-acrylic': np.float64(0.011314874772763571), 'special-walnut-224': np.float64(0.002932466447364288), 'green-metallic-paint': np.float64(0.0044852645193129695), 'green-plastic': np.float64(0.012508190703453114), 'pink-plastic': np.float64(0.0029386390627326353), 'red-plastic': np.float64(0.0034235232819379625), 'red-fabric': np.float64(0.0009153039440968264), 'maroon-plastic': np.float64(0.012859067679396432), 'white-marble': np.float64(0.012114085688997876), 'green-latex': np.float64(0.0009609997960958653), 'black-oxidized-steel': np.float64(0.0035264101465382063), 'natural-209': np.float64(0.004237699265320948), 'pearl-paint': np.float64(0.004014653875097759), 'white-diffuse-bball': np.float64(0.0047381524343442115), 'pink-fabric': np.float64(0.0011018708230415212), 'ipswich-pine-221': np.float64(0.0039425360376397645), 'blue-rubber': np.float64(0.0035778712176121946), 'tungsten-carbide': np.float64(0.01733837389935382), 'gold-paint': np.float64(0.004137414600145732), 'nylon': np.float64(0.006080186924030818), 'silver-metallic-paint2': np.float64(0.00545883700745851), 'pickled-oak-260': np.float64(0.0029599852833010116), 'specular-yellow-phenolic': np.float64(0.013449335854018203), 'specular-blue-phenolic': np.float64(0.0139828538624353), 'silver-paint': np.float64(0.004515969468440276), 'specular-violet-phenolic': np.float64(0.013648027372679328), 'black-obsidian': np.float64(0.013497781900115618), 'specular-black-phenolic': np.float64(0.013367349369820635), 'blue-metallic-paint': np.float64(0.004211661549866233), 'gold-metallic-paint2': np.float64(0.009344605904973364), 'neoprene-rubber': np.float64(0.003985335107783488), 'blue-acrylic': np.float64(0.012791747981517474), 'polyethylene': np.float64(0.00390707372649572), 'fruitwood-241': np.float64(0.00417633763783416), 'green-fabric': np.float64(0.001673829270629704), 'specular-red-phenolic': np.float64(0.013612333208512996), 'color-changing-paint2': np.float64(0.00469630996089597), 'two-layer-silver': np.float64(0.008446795465448555), 'gold-metallic-paint3': np.float64(0.0037332931431036877), 'alum-bronze': np.float64(0.0038152040125845753), 'blue-metallic-paint2': np.float64(0.0027046065781399856), 'red-phenolic': np.float64(0.011503889859280159), 'dark-red-paint': np.float64(0.002807225300251947), 'purple-paint': np.float64(0.0035255440284764337), 'delrin': np.float64(0.004689113144682743), 'green-acrylic': np.float64(0.013116174142837762), 'silicon-nitrade': np.float64(0.014808602849601827), 'blue-fabric': np.float64(0.0006079432359370366), 'specular-orange-phenolic': np.float64(0.012487673172697389), 'red-specular-plastic': np.float64(0.012162138179427455), 'chrome': np.float64(0.017571960550990997), 'orange-paint': np.float64(0.002928044199349707), 'white-fabric2': np.float64(0.0005186966834054653), 'cherry-235': np.float64(0.0038193964239328433), 'color-changing-paint1': np.float64(0.0121962028263665)}
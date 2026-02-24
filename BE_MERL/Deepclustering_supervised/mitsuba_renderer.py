import numpy as np
import bsdf
import mitsuba as mi


# ---------------------------------------------------------
# Lecture MERL
# ---------------------------------------------------------

def read_merl(filename):

    with open(filename, "rb") as f:

        dims = np.fromfile(f, dtype=np.int32, count=3)

        theta_h, theta_d, phi_d = dims

        n = theta_h * theta_d * phi_d * 3

        data = np.fromfile(f, dtype=np.float64, count=n)

    data = data.reshape(theta_h, theta_d, phi_d, 3)

    return np.ascontiguousarray(data.astype(np.float32))


# ---------------------------------------------------------
# Conversion BSDF
# ---------------------------------------------------------

def convert_merl_to_bsdf(merl_file, bsdf_file):

    data = read_merl(merl_file)

    theta_h, theta_d, phi_d, _ = data.shape

    obj = {
        "type": "bsdf",
        "format": "merl",
        "theta_h": int(theta_h),
        "theta_d": int(theta_d),
        "phi_d": int(phi_d),
        "values": data
    }

    with open(bsdf_file, "wb") as f:

        f.write(bsdf.dumps(obj))


    print("BSDF file written:", bsdf_file)


# ---------------------------------------------------------
# Vérification BSDF brut
# ---------------------------------------------------------

def verify_bsdf(bsdf_file):

    with open(bsdf_file, "rb") as f:

        obj = bsdf.loads(f.read())

    print("\nBSDF content:")
    print("type:", obj["type"])
    print("shape:", obj["values"].shape)
    print("dtype:", obj["values"].dtype)


# ---------------------------------------------------------
# Vérification Mitsuba
# ---------------------------------------------------------

def verify_mitsuba(bsdf_file):

    import mitsuba as mi
    mi.set_variant("scalar_rgb")

    bsdf = mi.load_dict({
        "type": "measured",
        "filename": bsdf_file
    })

    print(bsdf)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    merl_input = "steel.binary"
    bsdf_output = "steel.bsdf"

    convert_merl_to_bsdf(merl_input, bsdf_output)

    verify_bsdf(bsdf_output)

    verify_mitsuba(bsdf_output)

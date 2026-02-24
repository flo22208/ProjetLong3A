import argparse
import numpy as np

from merlDB.rendering import Renderer
from merlDB import database as db
import time
import random




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merldir', default='merlDB/db/brdfs/')
    parser.add_argument('--outdir', default='temp/frames/')
    parser.add_argument('--mat', default=None)
    args = parser.parse_args()


    renderer = Renderer(size=1000, save_path=args.outdir,nb_spheres=1,nb_tours=10,gamma=2.222)
    dbuilder = db.DBuilder(interp_method="linear",db_path=args.merldir)
    print(dbuilder.list_db())
    if not args.mat is None:
        mat = args.mat
    else:
        ldb = dbuilder.list_db()
        mat = ldb[random.randint(0,len(ldb)-1)]
    dbuilder.load_mat(mat)
    print("Loaded " + mat)
    brdf = dbuilder.brdf_function(mat)
    t1 = time.time()
    renderer.render(brdf, nb_images=10,name=mat,MP=4)
    t2 = time.time()
    print("Finished in", t2 -t1, "s")

import os
import json
from tqdm import tqdm

all_json = "../../X_KINETICS/data/pairs/all_images_train_100000.json"
paired_json = "../../X_KINETICS/data/pairs/paired_images_train_100000.json"

with open(all_json, 'r') as f:
    all_images = json.load(f)

all_images = all_images[0]

for key in tqdm(all_images):
    fname = all_images[key]
    new_fname = fname.replace(' ', '_')
    if os.path.exists(fname):
        os.rename(fname, new_fname)
    all_images[key] = new_fname

with open(paired_json, 'r') as f:
    paired_images = json.load(f)

paired_images = paired_images[0]

for key in tqdm(paired_images):
    fname1 = paired_images[key]['image1']
    new_fname1 = fname1.replace(' ', '_')
    if os.path.exists(fname1):
        os.rename(fname1, new_fname1)
    elif os.path.exists(new_fname1):
        pass
    paired_images[key]['image1'] = new_fname1

    fname2 = paired_images[key]['image2']
    new_fname2 = fname2.replace(' ', '_')
    if os.path.exists(fname2):
        os.rename(fname2, new_fname2)
    elif os.path.exists(new_fname2):
        pass
    paired_images[key]['image2'] = new_fname2

with open(all_json, 'w') as f:
    json.dump([all_images], f)

with open(paired_json, 'w') as f:
    json.dump([paired_images], f)

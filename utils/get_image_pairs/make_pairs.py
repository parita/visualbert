import argparse
import cv2
import os
from tqdm import tqdm
import json


def load_dataset():
    if args.dataset_name == 'vlog':
        from dataset import vlog_dataset
        return vlog_dataset(args.data_dir, time_step = 5, split_file = args.split,
                            dataset_name = args.dataset_name, mode = args.mode)
    else:
        from dataset import dataset
        return dataset(args.data_dir, time_step = 5, split_file = args.split,
                       dataset_name = args.dataset_name, mode = args.mode)

def save_images(itr, batch, paired_images_json, all_images_json):
    images = batch["images"]
    image_ids = batch["image_ids"]

    image_dir = os.path.join(args.save_dir, 'data', "{}_{}".format(args.mode, args.num_pairs))

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    out_files = [os.path.join(image_dir, i + '.jpg') for i in image_ids]

    for i, img in enumerate(images):
        cv2.imwrite(out_files[i], images[i])
        all_images_json[itr * 2 + i] = out_files[i]

    pair_info = {}
    pair_info["label"] = batch["label"]
    pair_info["image1"] = out_files[0]
    pair_info["image2"] = out_files[1]

    paired_images_json[itr] = pair_info

def save_images_parallel(itr, batch, paired_images_json, all_images_json):
    images = batch["images"]
    image_ids = batch["image_ids"]

    image_dir = os.path.join(args.save_dir, 'data', "{}_{}".format(args.mode, args.num_pairs))

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    out_files = [os.path.join(image_dir, i + '.jpg') for i in image_ids]

    for i, img in enumerate(images):
        cv2.imwrite(out_files[i], images[i])
        all_images_json[itr * 2 + i] = out_files[i]

    pair_info = {}
    pair_info["label"] = batch["label"]
    pair_info["image1"] = out_files[0]
    pair_info["image2"] = out_files[1]

    paired_images_json[itr] = pair_info


def create_pairs_parallel(dataset, rangeList):
    dataset_len = dataset.__len__()

    all_images_json = {}
    paired_images_json = {}

    itr = 0
    pbar = tqdm(total = args.num_pairs)
    for itr in rangeList:
        index = itr % dataset_len
        batch = dataset.__getitem__(index)
        save_images(itr, batch, paired_images_json, all_images_json)
        itr += 1
        pbar.update(1)
    pbar.close()

    with open(args.all_images_file, 'w') as f:
        json.dump([all_images_json], f)

    with open(args.paired_images_file, 'w') as f:
        json.dump([paired_images_json], f)

def create_pairs(dataset, rangeList = None):
    dataset_len = dataset.__len__()

    # num_processes = multiprocessing.cpu_count() // 4

    all_images_json = {}
    paired_images_json = {}

    itr = 0
    pbar = tqdm(total = args.num_pairs)
    while itr < args.num_pairs:
        index = itr % dataset_len
        batch = dataset.__getitem__(index)
        save_images(itr, batch, paired_images_json, all_images_json)
        itr += 1
        pbar.update(1)
    pbar.close()

    with open(args.all_images_file, 'w') as f:
        json.dump([all_images_json], f)

    with open(args.paired_images_file, 'w') as f:
        json.dump([paired_images_json], f)


def main():
    global args

    ap = argparse.ArgumentParser(description = "Make positive/negative pairs from VLOG dataseT")
    ap.add_argument("--data_dir", type = str, required = True)
    ap.add_argument("--save_dir", type = str, required = True)
    ap.add_argument("--mode", type = str, required = True)
    ap.add_argument("--split", type = str, default = None)
    ap.add_argument("--num_pairs", type = int, default = 1000)
    ap.add_argument("--dataset_name", type = str, default = 'kinetics')

    args = ap.parse_args()

    args.all_images_file = os.path.join(
        args.save_dir, "all_images_{}_{}.json".format(args.mode, args.num_pairs))
    args.paired_images_file = os.path.join(
        args.save_dir, "paired_images_{}_{}.json".format(args.mode, args.num_pairs))

    args.save_dir = os.path.abspath(args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = load_dataset()

    create_pairs(dataset)

if __name__=="__main__":
    main()

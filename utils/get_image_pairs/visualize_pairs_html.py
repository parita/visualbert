import argparse
import os
import json
from html4vision import Col, imagetable

def get_relative_image_path(path):
    dataset_folder = 'X_{}/'.format(args.dataset.upper())
    path = './' + path.split(dataset_folder)[-1]
    print(path)
    return path

def generate_html(pairs):
    image1_list = []
    image2_list = []
    labels_list = []
    for index in pairs:
        info = pairs[index]
        image1_list.append(get_relative_image_path(info['image1']))
        image2_list.append(get_relative_image_path(info['image2']))
        labels_list.append(str(info['label']))


    cols = [
        Col('id1', 'ID'),
        Col('img', 'Image 1', image1_list),
        Col('img', 'Image 2', image2_list),
        Col('text', 'Label', labels_list)
    ]

    imagetable(cols, out_file = 'index.html', imsize = (320, 240))

def main():
    global args

    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_file", required = True)
    ap.add_argument("--dataset", type = str, default = 'vlog')

    args = ap.parse_args()

    with open(args.pairs_file, 'r') as f:
        pairs = json.load(f)

    pairs = pairs[0]

    generate_html(pairs)

if __name__=='__main__':
    main()

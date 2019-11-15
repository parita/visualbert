import os
import random
import json
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import numpy
import torch
from torch.utils.data import Dataset
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
from copy import deepcopy

from torch.utils.data.dataloader import default_collate
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from pytorch_pretrained_bert.fine_tuning import _truncate_seq_pair, random_word
from dataloaders.bert_field import IntArrayField
from allennlp.data.fields import ListField

from .bert_data_utils import *
from visualbert.pytorch_pretrained_bert.tokenization import BertTokenizer

from PIL import Image

class KineticsDataset(Dataset):
    def __init__(self, args, visual_genome_chunk = False):
        super(KineticsDataset, self).__init__()
        self.args = args
        self.annots_path = args.annots_path
        self.split_name = args.split_name
        self.data_root = args.data_root
        self.visual_genome_chunk = visual_genome_chunk
        self.masks = args.masks

        self.image_feature_type = args.image_feature_type
        self.text_only = args.get("text_only", False)
        self.visual_only = args.get("visual_only", False)
        self.add_spatial_features = args.get("add_spatial_features", False)
        self.expanded = False
        ########## Loading JSON metadata file
        with open(self.annots_path, 'r') as f:
            self.items = json.load(f)[0]

        self.image_feat_reader = faster_RCNN_feat_reader()
        self.image_screening_parameters = self.args.image_screening_parameters

        """
        if self.args.get("chunk_path", None) is not None:
            self.chunk = torch.load(args.chunk_path)
            # Filter invalid entries
            new_items = {}
            for (key, value) in tqdm(self.items.items()):
                image_id_1 = value['image1']
                image_id_2 = value['image2']
                if "npz" not in image_id_1:
                    image_id_1 += ".npz"
                    image_id_2 += ".npz"

                if os.path.basename(image_id_1) not in self.chunk.keys():
                    continue

                if os.path.basename(image_id_2) not in self.chunk.keys():
                    continue

                value["image1"] = image_id_1
                value["image2"] = image_id_2
                new_items[key] = value

            self.items = new_items
        """
        self.feature_path = args.feature_path

        new_items = {}
        for (key, value) in tqdm(self.items.items()):
            image_id_1 = value['image1']
            image_id_2 = value['image2']
            try:
                im1 = Image.open(image_id_1)
                im2 = Image.open(image_id_2)
                new_items[key] = value
            except:
                continue

        self.items = new_items

        self.item_keys = [k for k in self.items.keys()]

        """
        average = 0.0
        new_chunk = {}
        for image_id in tqdm(self.chunk.keys()):
            filename = os.path.join(args.feature_path, image_id + ".npz")
            image_feat_variable, image_boxes, confidence = self.chunk[image_id]
            if "npz" not in image_id:
                image_id += ".npz"
                feature_obj  = screen_feature(
                    image_feat_variable, image_boxes, confidence,
                    self.image_screening_parameters
                )
            else:
                feature_obj = screen_feature(
                    image_feat_variable, image_boxes, confidence,
                    self.image_screening_parameters
                )

            average += feature_obj[2]
            new_chunk[image_id] = filename
            #if not os.path.exists(filename):
            np.savez(filename, image_feat = feature_obj[0],
                               cls_boxes = feature_obj[1],
                               image_loc = feature_obj[2])

        self.chunk = new_chunk
        torch.save(self.chunk, args.chunk_path)

        print("{} features on average.".format(average/len(self.chunk)))
        """

        print("{} of items in total.".format(len(self.items)))

        self.do_lower_case = args.do_lower_case
        self.bert_model_name = args.bert_model_name
        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name, do_lower_case=self.do_lower_case)
        self.pretraining = args.pretraining
        self.masked_lm_prob = args.get("masked_lm_prob", 0.15)

        """
        if self.image_feature_type == "r2c":
            items = []
            counter = 0
            for i in self.items:
                if self.expanded and index >= self.train_size:
                    image_file_name = "COCO_val2014_{:0>12d}.jpg".format(i['image_id'])
                else:
                    image_file_name = "COCO_{}2014_{:0>12d}.jpg".format(self.split_name, i['image_id'])
                if isinstance(self.masks[image_file_name], dict):
                    items.append(i)
                else:
                    # For some images, the detector seems to have Null output. Thus we just skip them. This will not affect much.
                    counter += 1
            print("Discarded {} instances in {}.".format(counter, self.split_name))
            self.items = items
        """

    def get_image_masks_by_training_index(self, index, which_one):
        item_key = self.item_keys[index]
        item = self.items[item_key]

        if which_one == 0:
            image_file_name = os.path.basename(item['image1'])
        else:
            image_file_name = os.path.basename(item['image2'])

        if self.args.get("chunk_path", None) is not None:
            basename = os.path.basename(self.chunk[image_file_name])
            filename = os.path.join(self.feature_path, basename)
            masks = np.load(filename)
        else:
            masks = None

        return masks

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        """
        item_key = self.item_keys[index]
        item = self.items[item_key]
        sample = {}
        if not self.text_only:
            image_feat_variable_0, image_boxes_0, image_dim_variable_0 = \
                    self.get_image_features_by_training_index(index, 0)
            image_feat_variable_1, image_boxes_1, image_dim_variable_1 = \
                    self.get_image_features_by_training_index(index, 1)

            visual_embeddings_type_0 = np.zeros(image_feat_variable_0.shape[0])
            visual_embeddings_type_1 = np.ones(image_feat_variable_1.shape[0])

            visual_embeddings_type = np.concatenate(
                (visual_embeddings_type_0, visual_embeddings_type_1), axis = 0)
            image_feat_variable = np.concatenate(
                (image_feat_variable_0, image_feat_variable_1), axis = 0)
            image_dim_variable = image_dim_variable_0 + image_dim_variable_1

            image_feat_variable = torch.Tensor(image_feat_variable)
            image_dim_variable = torch.LongTensor(image_dim_variable)
            visual_embeddings_type = torch.LongTensor(visual_embeddings_type)

            sample["image_feat_variable"] = image_feat_variable
            sample["image_dim_variable"] = image_dim_variable
            sample["visual_embeddings_type"] = visual_embeddings_type

        if item.get("label", None) is not None:
            sample["label"] = torch.LongTensor([1 if item["label"] == True else 0])
        else:
            sample["label"] = torch.LongTensor([0])

        sample["next_image_label"] = sample["label"]
        sample["is_random_next"] = sample["label"]
        """
        """
        caption_a = item["caption"]
        imageID = item["image_id"]

        if self.expanded and index >= self.train_size:
            coco = self.coco_val
        else:
            coco = self.coco

        rest_anns = coco.loadAnns([i for i in coco.getAnnIds(imgIds=imageID) if i != item['id']])

        if self.args.get("two_sentence", True):
            if random.random() > 0.5:
                item_b = self.items[random.randint(0, len(self.items) - 1)]
                while item_b["image_id"] == imageID:
                    item_b = self.items[random.randint(0, len(self.items) - 1)]
                flag = False
            else:
                item_b = rest_anns[random.randint(0, len(rest_anns) - 1)]
                flag = True

            caption_b = item_b["caption"]
            subword_tokens_a = self.tokenizer.tokenize(caption_a)
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = subword_tokens_b, is_correct=flag, max_seq_length = self.max_seq_length)
        elif not self.args.get("no_next_sentence", False):
            if random.random() < self.args.false_caption_ratio:
                item_b = self.items[random.randint(0, len(self.items) - 1)]
                while item_b["image_id"] == imageID:
                    item_b = self.items[random.randint(0, len(self.items) - 1)]
                flag = False
            else:
                item_b = item
                flag = True

            caption_b = item_b["caption"]
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_b, text_b = None, is_correct=flag, max_seq_length = self.max_seq_length)
        else:
            caption_b = item["caption"]
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_b, text_b = None, is_correct=None, max_seq_length = self.max_seq_length)

        bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
                    example = bert_example,
                    tokenizer=self.tokenizer,
                    probability = self.masked_lm_prob)
        bert_feature.insert_field_into_dict(sample)
        """

        if (self.image_feature_type == 'kinetics-r2c'):
            return self.__getitem_detector__(index)
        else:
            return self.__getitem_image__(index)

    def __getimage_detector__(self, image_file_path, metadata):
        sample = {}
        ###################################################################
        # Most of things adapted from VCR
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        if '.npz' in image_file_path:
            image_file_path = os.path.splitext(image_file_path)[0]

        image = load_image(image_file_path)
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape
        ###################################################################
        # We will use all detections
        dets2use = np.arange(len(metadata['cls_boxes']))
        # [nobj, 14, 14]
        #segms = np.stack([make_mask(mask_size=14, box=metadata['cls_boxes'][i],
        #                            polygons_list=metadata['segms'][i]) for i in dets2use])

        boxes = np.array(metadata['cls_boxes'])
        # Possibly rescale them if necessary
        boxes /= img_scale

        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]

        """
        try:
            metadata['names'] = [i.split(" ")[1][1:-1] for i in metadata["names"]]
        except:
            pass
        obj_labels = [self.coco_obj_to_ind[metadata['names'][i]] for i in dets2use.tolist()]
        """

        obj_labels = metadata['objects']
        keep_boxes = np.where(obj_labels > 0)

        boxes = boxes[keep_boxes]
        obj_labels = [0] + list(obj_labels[keep_boxes])
        obj_labels = [int(a) for a in obj_labels]

        boxes = np.row_stack((window, boxes))
        #segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)

        #sample['segms'] = ArrayField(segms, padding_value=0)
        #sample['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])
        sample["objects"] = IntArrayField(np.array(obj_labels))

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()

        if np.amax(boxes[:, 2]) >= w or np.amax(boxes[:, 3]) >= h:
            scale_w = (w - 1) / np.amax(boxes[:, 2])
            scale_h = (h - 1) / np.amax(boxes[:, 3])
            scale = min(scale_w, scale_h)
            boxes *= scale
            #print(np.amax(boxes[:, 2]), w)
            #print(np.amax(boxes[:, 3]), h)


        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        sample['boxes'] = torch.Tensor(boxes)

        return image, sample

    def __getimage__(self, image_file_path):
        sample = {}
        ###################################################################
        # Most of things adapted from VCR
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        if '.npz' in image_file_path:
            image_file_path = os.path.splitext(image_file_path)[0]

        image = load_image(image_file_path)
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape
        ###################################################################
        # Consider the entire image as a whole detected box

        boxes = np.array([window])
        obj_labels = [0]

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()

        """
        if np.amax(boxes[:, 2]) >= w or np.amax(boxes[:, 3]) >= h:
            scale_w = (w - 1) / np.amax(boxes[:, 2])
            scale_h = (h - 1) / np.amax(boxes[:, 3])
            scale = min(scale_w, scale_h)
            boxes *= scale
        """

        sample["objects"] = IntArrayField(np.array(obj_labels))
        sample['boxes'] = torch.Tensor(boxes)

        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))

        return image, sample


    def __getitem_image__(self, index):
        item_key = self.item_keys[index]
        item = self.items[item_key]
        sample = {}

        image_file_name_0 = item['image1']
        image_file_name_1 = item['image2']

        # masks_0 = self.get_image_masks_by_training_index(index, 0)
        # masks_1 = self.get_image_masks_by_training_index(index, 1)

        image_0, sample_0 = self.__getimage__(image_file_name_0)
        image_1, sample_1 = self.__getimage__(image_file_name_1)

        image = torch.stack((image_0, image_1), dim = 0)

        if item.get("label", None) is not None:
            sample["next_image_label"] = np.array([1 if item["label"] == True else 0])
        else:
            sample["next_image_label"] = np.array([0])

        sample["boxes_0"] = ArrayTensorField(sample_0["boxes"])
        sample["boxes_1"] = ArrayTensorField(sample_1["boxes"])

        sample["objects_0"] = sample_0["objects"]
        sample["objects_1"] = sample_1["objects"]

        sample["next_image_label"] = IntArrayField(sample["next_image_label"])
        sample["is_random_next"] = sample["next_image_label"]


        return image, Instance(sample)


    def __getitem_detector__(self, index):
        #print("Getting image bounding boxes using pre-trained detector")
        item_key = self.item_keys[index]
        item = self.items[item_key]
        sample = {}

        image_file_name_0 = item['image1']
        image_file_name_1 = item['image2']

        masks_0 = self.get_image_masks_by_training_index(index, 0)
        masks_1 = self.get_image_masks_by_training_index(index, 1)

        image_0, sample_0 = self.__getimage_detector__(image_file_name_0, masks_0)
        image_1, sample_1 = self.__getimage_detector__(image_file_name_1, masks_1)

        image = torch.stack((image_0, image_1), dim = 0)

        sample["boxes_0"] = ArrayTensorField(sample_0["boxes"])
        sample["boxes_1"] = ArrayTensorField(sample_1["boxes"])

        sample["objects_0"] = sample_0["objects"]
        sample["objects_1"] = sample_1["objects"]

        if item.get("label", None) is not None:
            sample["next_image_label"] = np.array([1 if item["label"] == True else 0])
        else:
            sample["next_image_label"] = np.array([0])

        sample["next_image_label"] = IntArrayField(sample["next_image_label"])
        sample["is_random_next"] = sample["next_image_label"]

        return image, Instance(sample)

    @classmethod
    def splits(cls, args):
        data_root = args.data_root

        if args.image_feature_type == "r2c":
            # For r2c, the masks are pre-computed from a larger detector. Thus, when pre-training on COCO, we follow the same procedure.

            masks = torch.load(os.path.join(data_root, "mask_train.th"))
            mask_val = torch.load(os.path.join(data_root, "mask_val.th"))
            for i in mask_val:
                masks[i] = mask_val[i]
        else:
            masks = None

        if args.image_feature_type == "flickr":
            import base64
            import csv
            import sys
            import zlib
            import time
            import mmap
            csv.field_size_limit(sys.maxsize)
            FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
            infiles = [
            os.path.join(data_root, "trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv"),
            os.path.join(data_root, "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0"),
            os.path.join(data_root, "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1"),
            os.path.join(data_root, "trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv")
            ]
            chunk = {}
            chunk_file = os.path.join(data_root, "trainval/resnet101_genome.th")
            if not os.path.exists(chunk_file):
                print("Loading COCO files for Flickr30K for the first time...")
                for infile in infiles:
                    with open(infile, "r+") as tsv_in_file:
                        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                        for item in tqdm(reader):
                            item['image_id'] = int(item['image_id'])
                            item['image_h'] = float(item['image_h'])
                            item['image_w'] = float(item['image_w'])
                            item['num_boxes'] = int(item['num_boxes'])
                            for field in ['boxes', 'features']:
                                # Hope the python2/3 b64decode does not mess things up.
                                item[field] = np.frombuffer(base64.b64decode(item[field]),
                                      dtype=np.float32).reshape((item['num_boxes'],-1))
                            item["features"] = torch.from_numpy(item["features"])
                            item["boxes"] = torch.from_numpy(item["boxes"])
                            chunk[item['image_id']] = item
                torch.save(chunk, chunk_file)
            else:
                chunk = torch.load(chunk_file)
        else:
            chunk = None

        copy_args = deepcopy(args)
        copy_args.split_name = "train"
        copy_args.annots_path = os.path.join(
            data_root, "data/pairs/paired_images_{}_1000.json".format(copy_args.split_name)
        )
        copy_args.feature_path = os.path.join(data_root, "features",
                                              "train", "npz_files")

        if args.image_feature_type == "kinetics-r2c":
            copy_args.chunk_path = os.path.join(
                data_root, "features", "train",
                "features_{}_150.th".format(copy_args.split_name)
            )

        if args.image_feature_type == "nlvr":
            copy_args.chunk_path = os.path.join(data_root, "coco_features_{}_150.th".format(copy_args.split_name))

        copy_args.data_root = data_root
        copy_args.masks = masks

        trainset = cls(copy_args, chunk)
        trainset.is_train = True

        copy_args = deepcopy(args)
        copy_args.split_name = "val"

        copy_args.annots_path = os.path.join(
            data_root, "data/pairs/paired_images_{}_100.json".format(copy_args.split_name)
        )
        copy_args.feature_path = os.path.join(data_root, "features",
                                              "val", "npz_files")

        if args.image_feature_type == "kinetics":
            copy_args.chunk_path = os.path.join(
                data_root, "features", "val",
                "features_{}_150.th".format(copy_args.split_name)
            )

        copy_args.data_root = data_root
        copy_args.masks = masks

        validationset = cls(copy_args, chunk)
        validationset.is_train = False

        """
        if args.get("expand_coco", False):
            # This is to expand the COCO train
            trainset.expanded = True
            trainset.train_size = len(trainset.items)

            trainset.items.extend(validationset.items)

            trainset.coco_val = validationset.coco

            if args.image_feature_type != "r2c" and args.image_feature_type != "vqa_fix_100" and args.image_feature_type != "flickr": # For NLVR, we pre-load features so we need to expand the chunk as well
                trainset.chunk_val = validationset.chunk

            imdb = np.load(os.path.join(data_root, "data/imdb/imdb_minival2014.npy"), allow_pickle = True)[1:]
            image_names_mini_val = set([i["image_name"] + ".jpg" for i in imdb])

            if args.get("exclude_minival", False):
                trainset.items = [i for i in trainset.items if "COCO_val2014_{:0>12d}.jpg".format(i['image_id']) not in image_names_mini_val]

            validationset.items = [i for i in validationset.items if "COCO_val2014_{:0>12d}.jpg".format(i['image_id']) in image_names_mini_val]
            print("After expanding, train has {} items, val has {} items".format(len(trainset.items), len(validationset.items)))
        """

        testset = validationset # Testset will not be used so this is just a placeholder
        return trainset, validationset, testset

    """
    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], dict):
            for index, i in enumerate(data):
                if "image_feat_variable" in i:
                    i["image_feat_variable"] = ArrayTensorField(i["image_feat_variable"])
                    i["image_dim_variable"] = IntArrayTensorField(i["image_dim_variable"])
                    i["visual_embeddings_type"] = IntArrayTensorField(i["visual_embeddings_type"])

                #i["bert_input_ids"] = IntArrayTensorField(i["bert_input_ids"])
                #i["bert_input_mask"] = IntArrayTensorField(i["bert_input_mask"])
                #i["bert_input_type_ids"] = IntArrayTensorField(i["bert_input_type_ids"])

                if "masked_lm_labels" in i:
                    i["masked_lm_labels"] = IntArrayTensorField(i["masked_lm_labels"], padding_value = -1)
                if "is_random_next" in i:
                    i["is_random_next"] = IntArrayTensorField(i["is_random_next"])

                i["next_image_label"] = IntArrayTensorField(i["next_image_label"])
                i['label'] = IntArrayTensorField(i['label'])

                data[index] = Instance(i)

        batch = Batch(data)
        td = batch.as_tensor_dict()
        td["label"] = td["label"].squeeze(-1)
        return td

    """

    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], Instance):
            batch = Batch(data)
            td = batch.as_tensor_dict()
            return td
        else:
            images, instances = zip(*data)
            images = torch.stack(images, 0)
            #boxes = torch.stack(boxes, 0)

            batch = Batch(instances)
            td = batch.as_tensor_dict()
            td['box_mask_0'] = torch.all(td['boxes_0'] >= 0, -1).long()
            td['box_mask_1'] = torch.all(td['boxes_1'] >= 0, -1).long()
            #td['objects'] = torch.stack((td['objects_0'], td['objects_1']), dim = 1)
            td['objects'] = None
            #del td['objects_0']
            #del td['objects_1']
            td['label'] = td['objects']
            td['images'] = images
            return td

# Modifed from R2C
"""
Dataloaders for VCR
"""
import json
import pickle
import os
import collections
import numpy as np
import numpy
import torch
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
from tqdm import tqdm
import cv2
from torchvision import transforms

from .vcr_data_utils import data_iter, data_iter_test, data_iter_item

from .bert_data_utils import InputExample, InputFeatures, get_one_image_feature_npz_screening_parameters, get_image_feat_reader, faster_RCNN_feat_reader, screen_feature

from .bert_field import IntArrayField

from visualbert.pytorch_pretrained_bert.fine_tuning import _truncate_seq_pair, random_word
from visualbert.pytorch_pretrained_bert.tokenization import BertTokenizer

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }

VIDEO_EXTENSIONS = ('.avi', '.mp4')

def is_video_file(f):
    return f.lower().endswith(VIDEO_EXTENSIONS)

def video_to_tensor(vid):
    return torch.from_numpy(vid.transpose([3, 0, 1, 2]))

class MPIIDataset(Dataset):
    def __init__(self, args):
        super(MPIIDataset, self).__init__()

        self.args = args
        self.annots_path = args.annots_path
        self.split_name = args.split_name
        self.data_root = args.data_root

        # Map each video file name to its annotated description
        self.annots = self.parse_annots(self.annots_path)

        # Map each video file name to its absolute location
        self.movies = self.parse_video_files(self.data_root)

        self.annots = {key:value for key, value in self.annots.items()
                       if key in self.movies.keys()}
        self.movies = {key:value for key, value in self.movies.items()
                       if key in self.annots.keys()}

        self.vocab = Vocabulary()

        self.do_lower_case = args.do_lower_case
        self.bert_model_name = args.bert_model_name

        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name, do_lower_case=self.do_lower_case)
        self.pretraining = args.pretraining

        # This is for pretraining
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = 20
        self.max_video_frames = 12

        self.rows = 224
        self.cols = 224

        self.transform_frame = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((self.rows, self.cols)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def parse_video_files(self, data_root):
        movies = collections.defaultdict()
        for root, _, files in os.walk(data_root):
            for f in files:
                if (is_video_file(f)):
                    filename = os.path.abspath(os.path.join(root, f))
                    movie_name = os.path.basename(os.path.dirname(filename))
                    movies[os.path.splitext(f)[0]] = {"filename": filename, "movie_name": movie_name}

        return movies

    def parse_annots(self, annots_path):
        annots = collections.defaultdict()
        with open(annots_path, 'r') as f:
            for line in f:
                line = line.split()
                filename, start_time, end_time, caption = line[0], line[1], line[2], line[-1]
                annots[filename] = {"start_time": start_time, "end_time": end_time, "caption": caption}
        return annots

    @classmethod
    def splits(cls, args):
        data_root = args.data_root

        copy_args = deepcopy(args)
        copy_args.split_name = "training"
        copy_args.annots_path = os.path.join(data_root, "LSMDC16_annos_{}.csv".format(copy_args.split_name))

        trainset = cls(copy_args)
        trainset.is_train = True

        copy_args = deepcopy(args)
        copy_args.split_name = "val"
        copy_args.annots_path = os.path.join(data_root, "LSMDC16_annos_{}.csv".format(copy_args.split_name))

        validationset = cls(copy_args)
        validationset.is_train = False

        testset = validationset

        return trainset, validationset, testset

    def __len__(self):
        return len(self.annots)

    def _get_video_frames(self, filename):
        if filename is not None:
            vc = cv2.VideoCapture(filename)
            length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = []
            count = 0
            return_val = True
            while count < length and return_val:
                return_val, frame = vc.read()
                frames.append(frame)
                count += 1

            vc.release()
        else:
            length = 0
            frames = []

        video_tensor = torch.zeros([3, self.max_video_frames, self.rows, self.cols])

        if len(frames) <= self.max_video_frames:
            num_frames = len(frames)
            indexes = range(len(frames))
        else:
            num_frames = self.max_video_frames
            indexes = range(len(frames))
            step = len(frames) // self.max_video_frames
            indexes = range(0, len(frames), step)[:self.max_video_frames]

        for vidx, fidx in enumerate(indexes):
            video_tensor[:, vidx, :, :] = self.transform_frame(frames[fidx])

        return np.array(num_frames), video_tensor

    def __getitem__(self, index):
        sample = {}

        all_keys = [k for k in self.annots.keys()]
        video_id_0 = all_keys[index]
        start_time_0 = self.annots[video_id_0]["start_time"]

        video_file_0 = self.movies[video_id_0]["filename"]
        movie_name_0 = self.movies[video_id_0]["movie_name"]

        video_id_1 = [vid for vid in self.annots.keys() \
                      if vid != video_id_0 \
                      and self.movies[vid]["movie_name"] == movie_name_0 \
                      and self.annots[vid]["start_time"] > start_time_0]

        if len(video_id_1):
            video_id_1 = np.random.choice(video_id_1)
            video_file_1 = self.movies[video_id_1]["filename"]
        else:
            video_id_1 = None

        if video_id_1 is not None and (self.args.get("next_video", True) or self.args.get("two_sentence", True)):
            if np.random.random() > 0.5:
                video_frames_0, video_0 = self._get_video_frames(video_file_0)
                video_frames_1, video_1 = self._get_video_frames(video_file_1)
                caption_0 = self.annots[video_id_0]["caption"]
                caption_1 = self.annots[video_id_1]["caption"]
                next_video_label = np.array([1])
            else:
                video_frames_1, video_1 = self._get_video_frames(video_file_0)
                video_frames_0, video_0 = self._get_video_frames(video_file_1)
                caption_1 = self.annots[video_id_0]["caption"]
                caption_0 = self.annots[video_id_1]["caption"]
                next_video_label = np.array([0])

            sample["video_0"] = ArrayField(video_0)
            sample["video_1"] = ArrayField(video_1)
            sample["video_frames_0"] = IntArrayField(video_frames_0)
            sample["video_frames_1"] = IntArrayField(video_frames_1)

            sample["next_video_label"] = IntArrayField(next_video_label)
            sample["is_random_next"] = IntArrayField(next_video_label)

            subword_tokens_0 = self.tokenizer.tokenize(caption_0)
            subword_tokens_1 = self.tokenizer.tokenize(caption_1)

            bert_example = InputExample(unique_id = index,
                                        text_a = subword_tokens_0, text_b = subword_tokens_1,
                                        is_correct = next_video_label,
                                        max_seq_length = self.max_seq_length)

        else:
            video_frames_0, video_0 = self._get_video_frames(video_file_0)
            video_frames_1, video_1 = self._get_video_frames(None)

            sample["video_0"] = ArrayField(video_0)
            sample["video_1"] = ArrayField(video_1)
            sample["video_frames_0"] = IntArrayField(video_frames_0)
            sample["video_frames_1"] = IntArrayField(video_frames_1)
            sample["next_video_label"] = IntArrayField(np.array([0]))
            sample["is_random_next"] = IntArrayField(np.array([0]))

            caption_0 = self.annots[video_id_0]["caption"]
            subword_tokens_0 = self.tokenizer.tokenize(caption_0)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_0,
                                        text_b = None, is_correct = None,
                                        max_seq_length = self.max_seq_length)

        bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
            example = bert_example, tokenizer = self.tokenizer,
            probability = self.masked_lm_prob)

        bert_feature.insert_field_into_dict(sample)

        return Instance(sample)

    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], Instance):
            batch = Batch(data)
            td = batch.as_tensor_dict()
            return td
        else:
            images, instances = zip(*data)
            images = torch.stack(images, 0)

            batch = Batch(instances)
            td = batch.as_tensor_dict()
            if 'question' in td:
                td['question_mask'] = get_text_field_mask(td['question'], num_wrapping_dims=1)
                td['question_tags'][td['question_mask'] == 0] = -2  # Padding
            if "answer" in td:
                td['answer_mask'] = get_text_field_mask(td['answers'], num_wrapping_dims=1)
                td['answer_tags'][td['answer_mask'] == 0] = -2

            td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
            td['images'] = images
            return td

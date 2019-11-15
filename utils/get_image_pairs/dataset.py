import os
import cv2
import logging
import argparse
import numpy as np
from tqdm import tqdm
from time import time

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

#from multiprocessing.pool import Pool
import multiprocessing
from multiprocessing import Process
from multiprocessing import JoinableQueue as Queue

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
VIDEO_EXTENSION = ['.mp4', '.mov']

logger = logging.getLogger(__name__)

class dataset(Dataset):
    def __init__(self, data_dir, time_step = 5, split_file = None, limit = None,
                 line_to_fname = lambda x: x, dataset_name = None, mode = None, **kwargs):

        self.limit = limit

        self.dataset_name = dataset_name
        if self.dataset_name == None:
            self.dataset_name = data_dir.strip('.').strip('/').replace('/', '-')

        self.data_dir = data_dir
        self.lists_dir = os.path.join('./data/lists', dataset_name)
        self.split_file = split_file
        self.mode = mode

        logger.debug("Dataset directory: {}".format(self.data_dir))

        self.loc = os.path.join(self.lists_dir, self.mode, str(self.limit))
        if not os.path.isdir(self.loc):
            os.makedirs(self.loc)

        self.time_step = time_step

        self.line_to_fname = line_to_fname
        self.parallel_processes = 32

        self._make_dataset()

    def _make_dataset(self):
        self.videos, self.video_lengths = self._parallely_make_dataset()
        self._integrate_lengths()

    def _parallely_make_dataset(self):
        #Get video_list and video_length files if they exist
        name_file = "{}/video_list.npy".format(self.loc)
        len_file = "{}/video_lengths.npy".format(self.loc)
        if os.path.isfile(name_file):
            video_list = np.load(name_file)
            video_lengths = np.load(len_file)
            return video_list, video_lengths

        #Files don't yet exist, so create them
        q = Queue()
        qvideo_list = Queue()

        #Collect list of videos to index
        fnames_list = []
        if self.split_file != None:
            with open(self.split_file, 'r') as f:
                line = f.readline()
                while line:
                    fnames_list.append(self.line_to_fname(os.path.join(self.data_dir, line.strip())))
                    line = f.readline()
        else:
            for root, _, fnames in tqdm(os.walk(self.data_dir)):
                for fname in sorted(fnames):
                    fnames_list.append(os.path.join(root, fname))

        #Truncate list if necessary
        if self.limit is not None:
            fnames_list = fnames_list[:self.limit]

        #Parallely open videos, get length
        def parallel_worker(fnames_chunk):
            item = q.get()
            for fname in tqdm(fnames_chunk):
                if has_file_allowed_extension(fname, VIDEO_EXTENSION):
                    video_path = fname
                    vc = cv2.VideoCapture(video_path)
                    length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
                    if length > 0 and vc.isOpened():
                        qvideo_list.put((video_path, length))
                        qvideo_list.task_done()
                    vc.release()
            q.task_done()

        processes = self.parallel_processes
        if self.limit is not None and processes >= self.limit:
            processes = self.limit

        n = len(fnames_list)
        chunk = int(n / processes)
        if chunk == 0:
            chunk = 1
        fnames_chunks = [fnames_list[i*chunk:(i+1)*chunk] \
                        for i in range((n + chunk - 1) // chunk)]
        for i in range(processes):
            q.put(i)
            multiprocessing.Process(target = parallel_worker,
                                    args = (fnames_chunks[i],)).start()

        q.join()
        qvideo_list.join()

        video_list = []
        video_lengths = []

        while qvideo_list.qsize() != 0:
            video, length = qvideo_list.get()
            video_list.append(video)
            video_lengths.append(length)

        np.save(name_file, video_list)
        np.save(len_file, video_lengths)

        return video_list, video_lengths

    def _integrate_lengths(self):
        # Save video_lengths as an integral array for faster lookups
        total = 0
        for idx in range(len(self.video_lengths)):
            total += self.video_lengths[idx] - self.time_step
            self.video_lengths[idx] = total

    def __len__(self):
        return len(self.videos)

    def _get(self, index):
        video_path = os.path.abspath(self.videos[index])
        vc = cv2.VideoCapture(video_path)
        length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

        if length <= 0:
            logger.error("Video {} is empty".format(video_path))
            return None

        if not vc.isOpened():
            logger.error("Failed to open video {}",format(video_path))
            return None

        if length < 5:
            logger.error("Video {} is not long enough to generate pair".format(video_path))
            return None

        images = []
        count = 0
        return_val = True
        while count < length and return_val:
            return_val, frame = vc.read()
            images.append(frame)
            for _ in range(self.time_step - 1):
                return_val, frame = vc.read()
            count += 1

        vc.release()

        video_name = os.path.splitext(video_path)[0]
        dataset_folder = "X_{}".format(self.dataset_name.upper())
        video_name = video_name.split("{}/data/".format(dataset_folder))[-1]
        video_name = "_".join(video_name.split("/"))
        video_name = video_name.replace(' ', '_')
        return images, video_name

    def sample_pair(self, videos, video_names, pos_prob = 0.5):
        pos_label = np.random.binomial(size = 1, n = 1, p = pos_prob)

        image_id_0 = "{}_{}".format(video_names[0], 0)
        image_0 = videos[0][0]

        if pos_label[0] == 1:
            max_step = min(len(videos[0]) // 2, 4)
            max_step = max(5, max_step)

            pos_idx = np.random.choice(range(1, max_step))
            image_1 = videos[0][pos_idx]
            image_id_1 = "{}_{}".format(video_names[0], pos_idx)
            label = True
        else:
            neg_idx = np.random.choice(range(0, len(videos[1])))
            image_1 = videos[1][neg_idx]
            image_id_1 = "{}_{}".format(video_names[1], neg_idx)
            label = False

        images = [image_0, image_1]
        image_ids = [image_id_0, image_id_1]

        return images, label, image_ids

    def __getitem__(self, index):
        video, video_name = self._get(index)
        next_video, next_video_name = self._get((index + 1) % self.__len__())
        images, label, image_ids = self.sample_pair([video, next_video], [video_name, next_video_name])

        #image_ids = ["{}_{}".format(video_name, image_id) for image_id in image_ids]

        output = {}
        output["images"] = images
        output["image_ids"] = image_ids
        output["label"] = label
        return output

class vlog_dataset(dataset):
    def __init__(self, *args, **kwargs):
        kwargs['line_to_fname'] = lambda x: os.path.join(x, 'clip.mp4')
        super(vlog_dataset, self).__init__(*args, **kwargs)

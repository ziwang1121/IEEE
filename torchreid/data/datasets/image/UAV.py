from __future__ import division, print_function, absolute_import
from os import pardir
import re
import glob
import os.path as osp
from torchreid import data

from numpy.lib.twodim_base import tri
from torchreid.data.datasets import image
import warnings

from ..dataset import MultiModalImageDataset


class UAV(MultiModalImageDataset):
    dataset_dir = 'UAVdata'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated.'
            )

        self.train_dir = osp.join(self.data_dir, 'reid_bounding_box_train', 'train')
        self.query_dir = osp.join(self.data_dir, 'reid_bounding_box_train', 'query')
        self.gallery_dir = osp.join(self.data_dir, 'reid_bounding_box_train', 'gallery')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(UAV, self).__init__(train, query, gallery, **kwargs)


    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        pid_container = set()
        for img_path in img_paths:
            jpg_name = img_path.split('\\')[-1]
            pid = int(jpg_name.split('.')[0][1:4])
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        i = 0
        for img_path in img_paths:
            jpg_name = img_path.split('\\')[-1]
            img = []
            img.append(img_path)
            img.append(osp.join(dir_path, 'Gray', jpg_name))
            pid = int(jpg_name.split('.')[0][1:4])
            i = i + 1
            camid = i
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img, pid, camid))
        return data




from __future__ import division, print_function, absolute_import
from os import pardir
import re
import glob
import os.path as osp

from numpy.lib.twodim_base import tri
from torchreid.data.datasets import image
import warnings

from ..dataset import MultiModalImageDataset


class AllDay(MultiModalImageDataset):
    dataset_dir = 'AllDay'

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


        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query_all')
        self.gallery_dir = osp.join(self.data_dir, 'gallery_all')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False)

        super(AllDay, self).__init__(train, query, gallery, **kwargs)


    def _process_dir_train(self, dir_path, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            jpg_name = img_path_RGB.split('/')[-1]
            pid = int(jpg_name.split('_')[0][0:6])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            img = []
            jpg_name = img_path_RGB.split('/')[-1]
            img_path_NI = osp.join(dir_path, 'NI', jpg_name)
            img_path_TI = osp.join(dir_path, 'TI', jpg_name)
            img.append(img_path_RGB)
            img.append(img_path_NI)
            img.append(img_path_TI)
            pid = int(jpg_name.split('_')[0][0:6])
            camid = int(jpg_name.split('_')[1][3])
            camid -= 1 # index starts from 0
            timeid = int(jpg_name.split('_')[2])
            # print(img, pid, camid, timeid)
            if relabel:
                pid = pid2label[pid]
            data.append((img, pid, camid, timeid))
            # print((img, pid, camid, timeid))
        return data
        
    def _process_dir_test(self, dir_path, relabel=False):
            img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
            pid_container = set()
            for img_path_RGB in img_paths_RGB:
                jpg_name = img_path_RGB.split('/')[-1]
                pid = int(jpg_name.split('_')[0][0:6])
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            data = []
            for img_path_RGB in img_paths_RGB:
                img = []
                jpg_name = img_path_RGB.split('/')[-1]
                img_path_NI = osp.join(dir_path, 'NI_change_ID_final', jpg_name)
                img_path_TI = osp.join(dir_path, 'TI_change_ID_final', jpg_name)
                img.append(img_path_RGB)
                img.append(img_path_NI)
                img.append(img_path_TI)
                pid = int(jpg_name.split('_')[0][0:6])
                camid = int(jpg_name.split('_')[1][3])
                camid -= 1 # index starts from 0
                timeid = int(jpg_name.split('_')[2])
                # print(img, pid, camid, timeid)
                if relabel:
                    pid = pid2label[pid]
                data.append((img, pid, camid, timeid))
            # print(data[0:10])
            return data

from __future__ import division, print_function, absolute_import
from operator import index
from pickle import PERSID
from random import random
import time
from torchreid.metrics import rank
from cv2 import data
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
from numpy.lib.type_check import real
import torch
import random
import matplotlib.pyplot as plt
from sklearn import manifold
import os

from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)
from torchreid.losses import DeepSupervision


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, use_gpu=True):
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.epoch = 0

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, mAP, save_dir, is_best=False):
        names = self.get_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models[name].state_dict(),
                    'epoch': epoch + 1,
                    'mAP': mAP,
                    'optimizer': self._optims[name].state_dict(),
                    'scheduler': self._scheds[name].state_dict()
                },
                osp.join(save_dir, name),
                is_best=is_best
            )


    def save_best_checkpoint(self, epoch_num, epoch_mAP, top_mAPList, top_epochList, path):
        if len(top_mAPList) < 2 and epoch_mAP not in top_mAPList:
            top_mAPList.append(epoch_mAP)
            top_epochList.append(epoch_num)
            self.save_model(epoch_num, epoch_mAP, path)
        else:
            min_mAP = min(top_mAPList)
            min_mAP_index = top_mAPList.index(min_mAP)
            if epoch_mAP > min_mAP and epoch_mAP not in top_mAPList:
                os.remove(osp.join(r"E:\code\AAAI2022", path, "model", 'model.pth.tar-' + str(top_epochList[min_mAP_index] + 1)))
                top_mAPList[min_mAP_index] = epoch_mAP
                top_epochList[min_mAP_index] = epoch_num
                self.save_model(epoch_num, epoch_mAP, path)


    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[-1]['lr']

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """
        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True'
            )
        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        # if self.writer is None:
        #     self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print('=> Start training')

        top_mAP_List = []
        top_mAP_epochList = []
        train_begin = time.time()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            epoch_begin = time.time()
            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )
            epoch_end = time.time() - epoch_begin
            train_end = time.time() - train_begin
            print("Epoch Time: {}\t Total Time: {}\n" .format(str(datetime.timedelta(seconds=int(epoch_end))), str(datetime.timedelta(seconds=int(train_end)))))
            # self.save_best_checkpoint(self.epoch, 11.11, top_mAP_List, top_mAP_epochList, save_dir)
            if (self.epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (self.epoch+1) % eval_freq == 0 \
               and (self.epoch + 1) != self.max_epoch:
                mAP = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                self.save_best_checkpoint(self.epoch, mAP, top_mAP_List, top_mAP_epochList, save_dir)

        # if self.max_epoch > 0:
        #     print('=> Final test')
        #     rank1 = self.test(
        #         dist_metric=dist_metric,
        #         normalize_feature=normalize_feature,
        #         visrank=visrank,
        #         visrank_topk=visrank_topk,
        #         save_dir=save_dir,
        #         use_metric_cuhk03=use_metric_cuhk03,
        #         ranks=ranks
        #     )
        #     self.save_model(self.epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % print_freq == 0:
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t' 'lr {lr:.6f}\n' 
                    # 'time {time}\t'
                    # 'eta {eta}\t'
                    '{losses}\t'
                    .format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        lr=self.get_current_lr(),
                        losses=losses
                    )
                )

            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar(
                    'Train/lr', self.get_current_lr(), n_iter
                )

            end = time.time()


        self.update_lr()

    def forward_backward(self, data):
        raise NotImplementedError

    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            rank1, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )

            if self.writer is not None:
                self.writer.add_scalar(f'Test/{name}/rank1', rank1, self.epoch)
                self.writer.add_scalar(f'Test/{name}/mAP', mAP, self.epoch)

        return mAP

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):  
        visualize_feature = False
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_, camids_, timeids_ = [], [], [], []
            for batch_idx, data in enumerate(data_loader):
                # print(data)
                imgs, pids, camids, timeids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    for i in range(len(imgs)):
                        imgs[i] = imgs[i].cuda()
                end = time.time()
                features = self.extract_features(imgs, timeids)
                batch_time.update(time.time() - end)
                features = features.cpu().clone()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
                timeids_.extend(timeids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            timeids_ = np.asarray(timeids_)
            return f_, pids_, camids_, timeids_



        print('Extracting features from query set ...')
        qf, q_pids, q_camids, q_timeids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids, g_timeids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        # compute for RGBNT201 
        print('Computing CMC and mAP for {}' .format(dataset_name))
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_timeids,
            g_timeids,
            use_metric_cuhk03=use_metric_cuhk03
        )
        

        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        print('\n')
        if visrank:
            # visualize_ranked_results(
            #     distmat,
            #     self.datamanager.fetch_test_loaders(dataset_name),
            #     self.datamanager.data_type,
            #     width=self.datamanager.width,
            #     height=self.datamanager.height,
            #     save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
            #     topk=visrank_topk
            # )
            
            draw_label = random.sample(range(1, 30), 6)
            # draw_label = [25, 27, 28, 29, 30]
            self.showPointMultiModal(qf, q_pids, draw_label, r"E:\reidCode\AAAI 2022-master\log\tsne")

        return cmc[0], mAP

    def compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def extract_features(self, input, timeids):
        return self.model(input, timeids)

    def relabel(self, list):
        list_new = []
        index_new = 0
        list_new.append(index_new)
        for i in range(1, len(list)):
            if list[i] != list[i-1]:
                index_new = index_new + 1 
            list_new.append(index_new)
        return list_new

    def showPointMultiModal(self, features, real_label, draw_label, save_path):
        save_path = os.path.join(save_path, str(draw_label) + ".jpg")
        print("Draw points of features to {}" .format(save_path))
        real_label = self.relabel(real_label)
        f_R = features[:, 0:768]
        f_N = features[:, 768:1536]
        f_T = features[:, 1536:2304]
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        features_R_tsne = tsne.fit_transform(f_R)
        features_N_tsne = tsne.fit_transform(f_N)
        features_T_tsne = tsne.fit_transform(f_T)
        COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
        MARKS = ['*', 'o', '^']
        features_R_min, features_R_max = features_R_tsne.min(0), features_R_tsne.max(0)
        features_R_norm = (features_R_tsne - features_R_min) / (features_R_max - features_R_min)
        features_N_min, features_N_max = features_N_tsne.min(0), features_N_tsne.max(0)
        features_N_norm = (features_N_tsne - features_N_min) / (features_N_max - features_N_min)
        features_T_min, features_T_max = features_T_tsne.min(0), features_T_tsne.max(0)
        features_T_norm = (features_T_tsne - features_T_min) / (features_T_max - features_T_min)
        plt.figure(figsize=(20, 20))
        for i in range(features_R_norm.shape[0]):
            if real_label[i] in draw_label:
                index = draw_label.index(real_label[i])
                plt.scatter(features_R_norm[i, 0], features_R_norm[i, 1], s=300, color=COLORS[index % 6], marker=MARKS[0], alpha=0.4)
                plt.scatter(features_N_norm[i, 0], features_N_norm[i, 1], s=300, color=COLORS[index % 6], marker=MARKS[1], alpha=0.4)
                plt.scatter(features_T_norm[i, 0], features_T_norm[i, 1], s=400, color=COLORS[index % 6], marker=MARKS[2], alpha=0.4)
        plt.savefig(save_path)
        plt.close()


    def parse_data_for_train(self, data):
        # print(data)
        imgs = data['img']
        pids = data['pid']
        timeids = data['timeid']
        return imgs, pids, timeids

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        timeids = data['timeid']
        return imgs, pids, camids, timeids

    def two_stepped_transfer_learning(
        self, epoch, fixbase_epoch, open_layers, model=None
    ):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(model, open_layers)
        else:
            open_all_layers(model)

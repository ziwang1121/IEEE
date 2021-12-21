from __future__ import division, print_function, absolute_import


from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss, multiModalMarginLoss, CenterLoss

from ..engine import *


class ImageCMTEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_m (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_m=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=3,
        weight_m=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageCMTEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_m >= 0 and weight_x >= 0
        assert weight_m + weight_x > 0
        self.weight_m = 1
        self.weight_x = 1
        self.weight_t = 1
        self.weight_c = 1

        self.criterion_t = TripletLoss(margin=3)
        # self.criterion_c = CenterLoss() 
        self.criterion_m = multiModalMarginLoss(margin=3)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].cuda()
            pids = pids.cuda()

        outputs, features_RGB, features_NI, features_TI, features_all = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features_all, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        # if self.weight_c > 0:
        #     loss_c = 0
        #     loss_c += self.criterion_c(features_RGB, pids)
        #     loss_c += self.criterion_c(features_TI, pids)
        #     loss_c += self.criterion_c(features_NI, pids)
        #     loss += self.weight_c * loss_c
        #     print(loss_c)
        #     raise RuntimeError
        #     loss_summary['loss_c'] = loss_c.item()

        if self.weight_m > 0:
            loss_m = 0
            loss_m += self.criterion_m(features_RGB, features_NI, pids)
            loss_m += self.criterion_m(features_RGB, features_TI, pids)
            loss_m += self.criterion_m(features_NI, features_TI, pids)
            loss += self.weight_m * loss_m
            loss_summary['loss_m'] = loss_m.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_summary

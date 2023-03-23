from __future__ import division, print_function, absolute_import


from torchreid.losses import CrossEntropyLoss

from torchreid.metrics.accuracy import accuracy

from ..engine import Engine


class MultiModalImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
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
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
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
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(MultiModalImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    ### softmax for three modal
    def forward_backward(self, data):
        imgs, pids, timeids = self.parse_data_for_train(data)

        if self.use_gpu:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].cuda()
            pids = pids.cuda()
        

        outputs_R, outputs_N, outputs_T = self.model(imgs)


        loss_R = self.compute_loss(self.criterion, outputs_R, pids)
        loss_N = self.compute_loss(self.criterion, outputs_N, pids)
        loss_T = self.compute_loss(self.criterion, outputs_T, pids)

        loss = loss_R + loss_N + loss_T
        # loss = loss_R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc_R = 0; acc_N = 0; acc_T = 0
        if isinstance(outputs_R, (tuple, list)):
            for i in range(len(outputs_R)):
                acc_R += accuracy(outputs_R[i], pids)[0]
                acc_N += accuracy(outputs_N[i], pids)[0]
                acc_T += accuracy(outputs_T[i], pids)[0]
                # print(acc)
            acc_R /= len(outputs_R)
            acc_N /= len(outputs_N)
            acc_T /= len(outputs_T)
        else:
            acc_R = accuracy(outputs_R, pids)[0]
            acc_N = accuracy(outputs_N, pids)[0]
            acc_T = accuracy(outputs_T, pids)[0]


        loss_summary = {
            'loss_all': loss.item(),
            'loss_R': loss_R.item(),
            'acc_R': acc_R.item(),
            'loss_N': loss_N.item(),
            'acc_N': acc_N.item(),
            'loss_T': loss_T.item(),
            'acc_T': acc_T.item(),
        }

        return loss_summary






class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
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
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
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
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids, camids = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            camids = camids.cuda()

        # print(pids, camids)
        # raise RuntimeError
        outputs = self.model(imgs, camids)
        # print(outputs[0].shape)
        loss = self.compute_loss(self.criterion, outputs, pids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': accuracy(outputs, pids)[0].item()
        }

        return loss_summary


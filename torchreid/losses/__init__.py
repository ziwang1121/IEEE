from __future__ import division, print_function, absolute_import

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .hcloss import hetero_loss
from .multi_modal_margin_loss_new import multiModalMarginLossNew
from .time_loss import time_loss

def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """

    loss = 0.
    for i in range(len(xs)):
        loss += criterion(xs[i], y)


    # loss = 0.
    # for i in range(len(xs)-1):
    #     loss += criterion(xs[i], y)
    # loss += criterion(xs[i+1], y) * 3

    # loss = 0.
    # if len(xs) == 4:
    #     for i in range(len(xs)-1):
    #         loss += criterion(xs[i], y)
    #     loss += criterion(xs[i+1], y) * 3
    # else:
    #     for i in range(len(xs)):
    #         loss += criterion(xs[i], y)

    # loss = 0.
    # for x in xs:
    #     loss += criterion(x, y)
    # loss /= len(xs)

    return loss

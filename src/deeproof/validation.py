import resource
from torch.autograd import Variable
import numpy as np
import logging
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import average_precision_score
from deeproof.metrics import best_f2_score
from deeproof.common import MAINLOG
logger = logging.getLogger(MAINLOG)


# cf https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def validate(epoch, valid_loader, model, loss_func):
    # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.

    model.eval()
    total_loss = 0
    predictions = []
    true_labels_binarized = []

    logger.info("Starting Validation")
    for batch_idx, (data, target, target_binarized) in enumerate(tqdm(valid_loader)):
        true_labels_binarized.append(target_binarized.numpy())

        # data, target = data.cuda(async=True), target.cuda(async=True)
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        target_binarized = Variable(target_binarized, volatile=True)

        pred = model(data)
        predictions.append(F.softmax(pred).data.numpy())

        total_loss += loss_func(pred, target).data[0]

    avg_loss = total_loss / len(valid_loader)

    predictions = np.vstack(predictions)
    true_labels_binarized = np.vstack(true_labels_binarized)

    score = average_precision_score(
        y_true=true_labels_binarized, y_score=predictions)

    logger.info(
        "===> Validation - Avg. loss: {:.4f}\tAP score: {:.4f}".format(avg_loss, score))
    return score, avg_loss

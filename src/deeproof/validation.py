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

def validate(epoch, valid_loader, model, loss_func, lb):
    # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.

    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    logger.info("Starting Validation")
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
        true_labels.append(target.numpy())

        # data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(
            target, volatile=True)

        pred = model(data)
        predictions.append(F.softmax(pred).data.numpy())

        total_loss += loss_func(pred, target).data[0]

    avg_loss = total_loss / len(valid_loader)

    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)

    score = average_precision_score(y_true=true_labels, y_score=predictions)
    logger.info("Corresponding tags\n{}".format(lb.classes_))

    logger.info(
        "===> Validation - Avg. loss: {:.4f}\tAP score: {:.4f}".format(avg_loss, score))
    return score, avg_loss

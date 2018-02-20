from torch.autograd import Variable
import numpy as np
import logging
import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from deeproof.common import MAINLOG
logger = logging.getLogger(MAINLOG)


def predict(test_loader, model):
    model.eval()
    predictions = []

    logger.info("Starting Prediction")
    for batch_idx, (data, _, _) in enumerate(tqdm(test_loader)):
        # data = data.cuda(async=True)
        data = Variable(data, volatile=True)

        pred = F.softmax(model(data))
        predictions.append(pred.data.cpu().numpy())

    predictions = np.vstack(predictions)

    logger.info("===> Raw predictions done")
    # logger.info(predictions)
    return predictions


def write_submission_file(predictions, ids, dir_path, id_model, best_score):
    result = pd.DataFrame(columns=['id', 1, 2, 3, 4])
    result['id'] = ids
    result[[1, 2, 3, 4]] = predictions

    logger.info("===> Final predictions done")
    # logger.info(result)

    fname_result = dir_path / (str(id_model) + '-final-pred-' + str(best_score) + '.csv')
    result.to_csv(fname_result.as_posix(), index=False)
    logger.info("Final predictions saved to {}".format(fname_result.as_posix()))

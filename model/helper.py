import os
import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        if prediction.shape != y.shape:
            y = crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


def compute_receptive_field(depth, kernel_size, downsample_factor):
    fov = 1
    downsample_factor_prod = 1
    # encoder
    for layer in range(depth - 1):
        # two convolutions, each adds (kernel size - 1 ) * current downsampling level
        fov = fov + 2 * (kernel_size - 1) * downsample_factor_prod
        # downsampling multiplies by downsample factor
        fov = fov * downsample_factor
        downsample_factor_prod *= downsample_factor
    # bottom layer just two convs
    fov = fov + 2 * (kernel_size - 1) * downsample_factor_prod

    # decoder
    for layer in range(0, depth - 1)[::-1]:
        # upsample
        downsample_factor_prod /= downsample_factor
        # two convolutions, each adds (kernel size - 1) * current downsampling level
        fov = fov + 2 * (kernel_size - 1) * downsample_factor_prod

    return fov


def plot_receptive_field(unet, npseed=10, path="nuclei_train_data"):
    ds = CellDataset(path)
    np.random.seed(npseed)
    img_tensor = ds[np.random.randint(len(ds))][0]

    img_arr = np.squeeze(img_tensor.numpy())
    print(img_arr.shape)
    fov = compute_receptive_field(unet.depth, unet.kernel_size, unet.downsample_factor)

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img_arr)  # , cmap='gray')

    # visualize receptive field
    xmin = img_arr.shape[1] / 2 - fov / 2
    xmax = img_arr.shape[1] / 2 + fov / 2
    ymin = img_arr.shape[0] / 2 - fov / 2
    ymax = img_arr.shape[0] / 2 + fov / 2
    color = "red"
    plt.hlines(ymin, xmin, xmax, color=color, lw=3)
    plt.hlines(ymax, xmin, xmax, color=color, lw=3)
    plt.vlines(xmin, ymin, ymax, color=color, lw=3)
    plt.vlines(xmax, ymin, ymax, color=color, lw=3)
    plt.show()


def compute_affinities(seg: np.ndarray, nhood: list):

    nhood = np.array(nhood)

    shape = seg.shape
    n_edges = nhood.shape[0]
    dims = nhood.shape[1]
    affinity = np.zeros((n_edges,) + shape, dtype=np.int32)

    for e in range(n_edges):
        affinity[
            e,
            max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
        ] = (
            (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ]
                == seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ]
            )
            * (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ]
                > 0
            )
            * (
                seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ]
                > 0
            )
        )

    return affinity


def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray, th: float = 0.5):
    """Function to evaluate a segmentation."""

    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    overlay = np.array([pred_labels_rel.flatten(), gt_labels_rel.flatten()])

    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1
    )
    overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}

    for l, c in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel, return_counts=True)

    pred_labels_count_dict = {}
    for l, c in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)

    # create iou table
    iouMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)

    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)
        iouMat[int(v), int(u)] = iou

    # remove background
    iouMat = iouMat[1:, 1:]

    # use IoU threshold th
    if num_matches > 0 and np.max(iouMat) > th:
        costs = -(iouMat > th).astype(float) - iouMat / (2 * num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouMat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred_labels - tp
    fn = num_gt_labels - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = tp / (tp + fp + fn)

    return precision, recall, accuracy

def validate(
    model, 
    loader, 
    #rfs,
    ncols,
    nrows,
    cropsize,
    loss_function, 
    metric,
    step = None, 
    tb_logger = None, 
    device = None,
    ):

    if device is None:
        # default to gpu if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode for validation
    model.eval()
    model.to(device)

    # set validation loss and metrix
    val_loss = 0
    val_metric = 0
    predictions = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            # set target type same as prediction type. If "RunTimeError: found dType Float but expected
            # Short" message, look here.  
            if y.dtype != pred.dtype:
                y = y.type(pred.dtype)
            val_loss += loss_function(pred, y).item()
            val_metric += metric(pred > 0.5, y).item()

            #pred = F.center_crop(img = pred, output_size = rfs)
            predictions.append(pred)

        # Stitch crops back together
        predictions = torch.cat(predictions, dim = 0).cpu().numpy()
        slide = predictions.shape[-1]
        stitched = predictions.reshape(ncols * slide, nrows * slide)

    val_loss = val_loss / len(loader)
    val_metric = val_metric / len(loader)

    return stitched, val_loss, val_metric

    # if tb_logger is not None:
    #     assert(
    #         step is not None
    #     ), "Need to know the current step to show validation results"

    #     tb_logger.add_scalar(tag = "val_loss", scalar_value = val_loss, global_step=step)
    #     tb_logger.add_scalar(tag = "val_metric", scalar_value = val_metric, global_step = step)

    #     tb_logger.add_images(tag = "val input image", img_tensor = x.to("cpu"), global_step = step)
    #     tb_logger.add_images(tag = "val target image", img_tensor = y.to("cpu"), global_step = step)
    #     tb_logger.add_images(tag = "val prediction", img_tensor = pred.to("cpu"), global_step = step)

    # print(
    #     "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
    #         val_loss, val_metric
    #     )
    # )
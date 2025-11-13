from matplotlib.pylab import indices
import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt

from typing import List, Tuple


def saliency_map(
    model: torch.nn.Module, img: torch.Tensor, target_class: int
) -> torch.Tensor:
    """Calculate the saliency map for a given image and target class.
    Args:
        model (torch.nn.Module): The trained model.
        img (torch.Tensor): The input image tensor of shape (1, 1, 28, 28).
        target_class (int): The target class index.
    Returns:
        torch.Tensor: The saliency map of shape (28, 28).
    """
    model.eval()
    img = img.clone().detach().requires_grad_(True)

    logits = model(img)
    score = logits[0, target_class]
    score.backward()

    sal = img.grad.detach().abs()[0, 0]

    return sal


def integrated_gradients(
    model: torch.nn.Module,
    img: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    steps: int = 50,
) -> torch.Tensor:
    """
    Compute Integrated Gradients for a given input and target class.

    Args:
        model (torch.nn.Module): a PyTorch model.
        img (torch.Tensor): input tensor with shape (1, C, H, W) or (1, D).
        target_class (int): index of the output class to attribute.
        baseline (torch.Tensor): reference input tensor (same shape as input). Defaults to zeros.
        steps (int): number of interpolation steps between baseline and input.
    Returns:
        (torch.Tensor): tensor of same shape as input, with feature attributions.
    """

    if baseline is None:
        baseline = torch.zeros_like(img)

    # Scale inputs and compute gradients
    scaled_inputs = [
        baseline + (float(i) / steps) * (img - baseline) for i in range(steps + 1)
    ]
    grads = []
    model.eval()
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        output = model(scaled_input)
        score = output[0, target_class]
        model.zero_grad()
        score.backward()
        grad = scaled_input.grad.detach()
        grads.append(grad)

    # Approximate the integral using the trapezoidal rule
    grads = torch.stack(grads)  # Shape: (steps + 1, 1, C, H, W) or (steps + 1, 1, D)
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    integrated_grads = (img - baseline) * avg_grads.mean(dim=0)

    return integrated_grads.squeeze()


def int_grad_alter_img(img, int_grad, modification_factor=0.1):
    int_grad_true_abs = int_grad.abs()
    threshold = torch.quantile(int_grad_true_abs, 1 - modification_factor)
    mask = int_grad_true_abs > threshold
    mask_signed = mask * torch.sign(int_grad)

    # Apply the mask
    # If it is a 1, convert all original pixel to white (1), if -1 convert to black (0), if 0 leave unchanged
    mask_expanded = mask_signed.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
    mask_expanded = mask_expanded.repeat(1, 3, 1, 1)  # shape (1,3,H,W)
    modified_img = img.clone()
    modified_img[mask_expanded == 1] = 1.0
    modified_img = torch.where(
        mask_expanded == -1, img[0, :, 0, 0].view(1, 3, 1, 1), modified_img
    )
    return modified_img, mask_signed


def get_all_integrated_gradients(
    model: torch.nn.Module,
    imgs: Tuple[List[torch.Tensor]],
    baseline: torch.Tensor = None,
    steps: int = 50,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Compute Integrated Gradients for all steps between baseline and input.

    Args:
        model (torch.nn.Module): a PyTorch model.
        imgs (Tuple[List[torch.Tensor]]): input tensors with shape (1, C, H, W) or (1, D).
        baseline (torch.Tensor): reference input tensor (same shape as input). Defaults to zeros.
        steps (int): number of interpolation steps between baseline and input.
    Returns:
        (List[torch.Tensor], torch.Tensor): Return list of integrated gradients for predicted class,
        true class and their difference for each image.
    """
    ig_preds = []
    ig_trues = []
    ig_difs = []

    for label, img in imgs:
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()

        ig_pred = integrated_gradients(model, img, pred, baseline, steps)
        ig_pred_gray = ig_pred.abs().sum(dim=0)

        ig_true = integrated_gradients(model, img, label, baseline, steps)
        ig_true_gray = ig_true.abs().sum(dim=0)

        ig_dif = ig_true_gray - ig_pred_gray

        ig_preds.append(ig_pred_gray)
        ig_trues.append(ig_true_gray)
        ig_difs.append(ig_dif)

    return ig_preds, ig_trues, ig_difs

import torch
import torch.nn.functional as F


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
):
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

    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(img)

    alphas = torch.linspace(0, 1, steps).unsqueeze(1).to(img.device)
    scaled_inputs = baseline + alphas * (img - baseline)

    grads = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.unsqueeze(0).requires_grad_(True)
        output = model(scaled_input)
        target = output[0, target_class]
        model.zero_grad()
        target.backward(retain_graph=True)
        grads.append(scaled_input.grad.detach())

    grads = torch.stack(grads).mean(dim=0)
    attributions = (img - baseline) * grads

    return attributions

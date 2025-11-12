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
    # compute saliency map
    model.eval()
    img.requires_grad_()

    logits = model(img)

    loss = F.cross_entropy(logits, torch.tensor([target_class], device=img.device))
    loss.backward()

    saliency, _ = img.grad.data.abs().max(dim=1)
    saliency = saliency.reshape(28, 28)
    return saliency


def targeted_fgsm(
    model: torch.nn.Module, img: torch.Tensor, target_class: int, eps: float = 0.1
):
    """Generate a targeted adversarial example using the Fast Gradient Sign Method (FGSM).

    Args:
        model (torch.nn.Module): The trained model.
        img (torch.Tensor): The input image tensor of shape (1, 1, 28, 28).
        target_class (int): The target class index.
        eps (float, optional): The perturbation magnitude. Defaults to 0.1.

    Returns:
        torch.Tensor: The adversarial example tensor.
    """
    model.eval()
    img_adv = img.clone().detach().requires_grad_(True)
    logits = model(img_adv)
    loss = F.cross_entropy(logits, torch.tensor([target_class], device=img.device))
    loss.backward()
    perturb = -eps * img_adv.grad.sign()
    adv = (img + perturb).clamp(0, 1).detach()
    return adv


def targeted_pgd(
    model: torch.nn.Module,
    img: torch.Tensor,
    target_class: int,
    eps: float = 0.2,
    alpha: float = 0.02,
    iters: int = 40,
):
    """Generate a targeted adversarial example using the Projected Gradient Descent (PGD) method.

    Args:
        model (torch.nn.Module): The trained model.
        img (torch.Tensor): The input image tensor of shape (1, 1, 28, 28).
        target_class (int): The target class index.
        eps (float, optional): The maximum perturbation. Defaults to 0.2.
        alpha (float, optional): The step size for each iteration. Defaults to 0.02.
        iters (int, optional): The number of iterations. Defaults to 40.

    Returns:
        torch.Tensor: The adversarial example tensor.
    """
    model.eval()
    adv = img.clone().detach()
    adv.requires_grad = True
    for i in range(iters):
        logits = model(adv)
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=img.device))
        loss.backward()
        adv = adv.detach() - alpha * adv.grad.sign()
        adv = torch.max(torch.min(adv, img + eps), img - eps)
        adv = adv.clamp(0, 1).detach()
        adv.requires_grad_(True)
    return adv.detach()


def sparse_targeted_opt(
    model: torch.nn.Module,
    img: torch.Tensor,
    target_class: int,
    saliency: torch.Tensor,
    k: int = 50,
    steps: int = 200,
    lr: float = 1e-1,
):
    """Generate a targeted adversarial example by optimizing perturbations only on the top-k salient pixels.

    Args:
        model (torch.nn.Module): The trained model.
        img (torch.Tensor): The input image tensor of shape (1, 1, 28, 28).
        target_class (int): The target class index.
        saliency (torch.Tensor): The saliency map tensor of shape (28, 28).
        k (int, optional): The number of top salient pixels to perturb. Defaults to 50.
        steps (int, optional): The number of optimization steps. Defaults to 200.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-1.

    Returns:
        torch.Tensor: The adversarial example tensor.
        torch.Tensor: The mask tensor indicating perturbed pixels.
    """
    device = img.device
    flat = saliency.view(-1)
    topk_idx = torch.topk(flat, k).indices
    mask = torch.zeros(28 * 28, device=device)
    mask[topk_idx] = 1.0
    mask = mask.view(1, 1, 28, 28)

    delta = torch.zeros_like(img, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for i in range(steps):
        adv = (img + delta * mask).clamp(0, 1)
        logits = model(adv)
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=device))
        # add small L2 regularizer to keep changes small
        loss_total = loss + 1e-3 * torch.norm((delta * mask).view(-1), p=2)
        opt.zero_grad()
        loss_total.backward()
        opt.step()
        delta.data = torch.clamp(delta.data, -0.5, 0.5)
        pred = logits.argmax(dim=1).item()
        if pred == target_class:
            break
    adv_final = (img + delta.data * mask).clamp(0, 1).detach()
    return adv_final, mask


def occlusion_sensitivity(
    model: torch.nn.Module,
    img: torch.Tensor,
    target_class: int,
    patch: int = 4,
    stride: int = 2,
):
    """Calculate the occlusion sensitivity heatmap for a given image and target class.

    Args:
        model (torch.nn.Module): The trained model.
        img (torch.Tensor): The input image tensor of shape (1, 1, 28, 28).
        target_class (int): The target class index.
        patch (int, optional): The size of the occlusion patch. Defaults to 4.
        stride (int, optional): The stride for moving the occlusion patch. Defaults to 2.

    Returns:
        torch.Tensor: The occlusion sensitivity heatmap tensor of shape (28, 28).
    """
    model.eval()
    _, _, H, W = img.shape
    heat = torch.zeros(H, W, device=img.device)
    base_pred = model(img).softmax(dim=1)[0, target_class].item()
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            img2 = img.clone()
            img2[:, :, y : y + patch, x : x + patch] = 0.0  # occlude
            p = model(img2).softmax(dim=1)[0, target_class].item()
            heat[y : y + patch, x : x + patch] += base_pred - p  # drop in prob
    return heat

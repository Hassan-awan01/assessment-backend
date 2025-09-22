# fgsm.py
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np

class Attack:
    """
    Implements FGSM attack for PyTorch models.
    """
    def __init__(self, model, device='cpu', loss_fn=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()

    def fgsm(self, input_tensor, true_label, epsilon=0.1):
        """
        input_tensor: torch.Tensor of shape (1, C, H, W), requires_grad=True
        true_label: torch.tensor shape (1,) with the label (int)
        returns: adversarial_tensor (clamped), and the perturbation
        """
        # Ensure model in eval
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        loss = self.loss_fn(output, true_label.to(self.device))
        loss.backward()

        # Collect the sign of the gradient on the input
        sign_data_grad = input_tensor.grad.data.sign()

        # Create the perturbed image by adjusting each pixel
        perturbed = input_tensor + epsilon * sign_data_grad

        perturbed = torch.clamp(perturbed, -1.0, 1.0)

        return perturbed.detach(), sign_data_grad.detach()

    @staticmethod
    def tensor_to_base64_image(tensor, unnormalize_mean=0.5, unnormalize_std=0.5):
        """
        Convert a single image tensor (1,C,H,W) or (C,H,W) in normalized range back to PNG base64.
        It assumes normalization transform = Normalize((mean,), (std,)).
        """
        t = tensor.detach().cpu()
        if t.dim() == 4:
            t = t[0]
        # unnormalize
        t = t * unnormalize_std + unnormalize_mean  # now in [0,1] approximately
        t = t.clamp(0, 1)
        # convert to PIL
        np_img = (t.numpy() * 255).astype(np.uint8)
        # for grayscale, shape (1, H, W)
        if np_img.shape[0] == 1:
            np_img = np_img[0]
            pil = Image.fromarray(np_img, mode='L')
        else:
            # (3,H,W) -> (H,W,3)
            pil = Image.fromarray(np.transpose(np_img, (1, 2, 0)))
        buffered = io.BytesIO()
        pil.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_b64

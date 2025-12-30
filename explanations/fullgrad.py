import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import FullGrad as FullGrad_

import utils


class FullGrad:
    """Class that implements FullGrad explanation method for a given model."""

    def __init__(self, model, device):
        """Initializes FullGrad_ object.

        Args:
            model (torch.nn.Module): Model to be explained.
            device (str): Device to be used ('cpu' or 'cuda').

        """
        super().__init__()

        self.device = device
        self.model = model

        # Check model type and set target layers accordingly, add a reshape transform if needed
        if model.name == "resnet50":
            target_layers = [model.resnet50[-2]]  # Last layer of ResNet50

            self.explainer = FullGrad_(model=model, target_layers=target_layers)

        elif model.name == "vitB":
            target_layers = [model.vit_b_16.encoder.layers.encoder_layer_11.ln_1]
            self.explainer = FullGrad_(model=model, target_layers=target_layers)

        elif model.name == "CLIP-zero-shot":
            self.model.preprocess = transforms.Compose(
                [transforms.ToTensor(), *self.model.preprocess.transforms]
            )
            target_layers = [model.clip_net.visual.transformer.resblocks[-2].ln_1]

            self.explainer = FullGrad_(model=model, target_layers=target_layers)

        else:
            print(f"Warning: FullGrad_ is not implemented for model {model.name}.")

    def compute_and_plot_explanation(self, image, **kwargs):
        """Compute FullGrad explanation and plot it.

        Args:
            image (PIL.Image): Input image.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            save_expl (str): Path to save the explanation.

        Returns:
            None
        """
        self.model.eval()
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        img_tensor = self.model.preprocess(image).to(self.device).unsqueeze(0).float()
        # Compute GradCAM explanation
        grayscale_cam = self.explainer(input_tensor=img_tensor)[0, :]
        rgb_img = np.array(image.resize((224, 224)))

        self.model.train()

        if save_expl:
            # Visualize the explanation and save it
            visualization = utils.show_cam_on_image(rgb_img / 255.0, grayscale_cam)
            visualization_pil = Image.fromarray(np.uint8(visualization))
            visualization_pil.save(save_expl)

        if save_activations:
            utils.save_as_npy(grayscale_cam, save_activations)

        if return_activations:
            return grayscale_cam
        return None

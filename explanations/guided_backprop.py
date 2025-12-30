import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM as GradCAM_
from pytorch_grad_cam import GuidedBackpropReLUModel as GuidedBackpropReLUModel_
from pytorch_grad_cam.utils.image import deprocess_image

import utils


class GuidedBackpropReLUModel:
    """Class that implements GuidedBackprop explanation method for a given model."""

    def __init__(self, model, device, dict_hyperparam=None):
        """Initializes GuidedBackprop object.

        Args:
            model (torch.nn.Module): Model to be explained.
            device (str): Device to be used ('cpu' or 'cuda').
            dict_hyperparam (dict): Dictionary of hyperparameters for GuidedBackprop. Defaults to None.

        """
        if dict_hyperparam is None:
            dict_hyperparam = {"pos_target_layer": [-2]}
        super().__init__()

        self.device = device
        self.model = model

        # Check model type and set target layers accordingly, add a reshape transform if needed
        if model.name == "resnet50":
            target_layers = [
                model.resnet50[int(pos)] for pos in dict_hyperparam["pos_target_layer"]
            ]  # Last layer of ResNet50
            self.gb_model = GuidedBackpropReLUModel_(model=model.eval(), device=device)
            self.explainer = GradCAM_(model=model.eval(), target_layers=target_layers)

        else:
            print(f"Warning: GuidedBackprop_ is not implemented for model {model.name}.")

    def compute_and_plot_explanation(self, image, **kwargs):
        """Compute GuidedBackprop explanation and plot it.

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

        # Compute GuidedBackprop explanation

        grayscale_cam = self.explainer(input_tensor=img_tensor)[0, :]

        gb = self.gb_model(img_tensor, target_category=None)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        deprocess_image(cam_mask * gb)

        gb_grayscale = gb[:, :, 0]
        gb_grayscale = gb_grayscale / (np.max(gb_grayscale) + 1e-5)

        rgb_img = np.array(image.resize((224, 224)))
        self.model.train()

        if save_expl:
            # Visualize the explanation and save it
            visualization = utils.show_cam_on_image(rgb_img / 255.0, gb_grayscale)
            visualization_pil = Image.fromarray(np.uint8(visualization))
            visualization_pil.save(save_expl)

        if save_activations:
            utils.save_as_npy(gb_grayscale, save_activations)

        if return_activations:
            return gb_grayscale
        return None

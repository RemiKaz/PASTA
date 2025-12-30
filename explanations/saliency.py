import numpy as np
from PIL import Image
from pytorch_grad_cam import ScoreCAM as ScoreCAM_

import utils


class ScoreCAM:
    """Class that implements ScoreCAM explanation method for a given model."""

    def __init__(self, model, device):
        """Initializes ScoreCAM object.

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

            self.explainer = ScoreCAM_(model=model, target_layers=target_layers)

        else:
            print(f"Warning: ScoreCAM is not implemented for model {model.name}.")

    def compute_and_plot_explanation(self, image, **kwargs):
        """Compute ScoreCAM explanation and plot it.

        Args:
            image (PIL.Image): Input image.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            save_expl (str): Path to save the explanation.

        Returns:
            None
        """
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        img_tensor = self.model.preprocess(image).to(self.device).unsqueeze(0).float()

        # Compute ScoreCAM explanation
        grayscale_cam = self.explainer(input_tensor=img_tensor)[0, :]

        rgb_img = np.array(image.resize((224, 224)))

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

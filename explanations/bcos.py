import numpy as np
from PIL import Image

import utils


class BCos:
    """Class that implements BCos explanation method for a given model."""

    def __init__(self, model, device):
        """Initializes BCos object.

        Args:
            model (torch.nn.Module): Model to be explained.
            device (str): Device to be used ('cpu' or 'cuda').

        """
        super().__init__()

        self.device = device
        self.model = model.to(device)

    def compute_and_plot_explanation(self, image, **kwargs):
        """Compute BCos explanation and plot it.

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

        # Compute BCos explanation
        self.model.eval()
        image_preprocess = self.model.preprocess(image)[None].to(self.device)

        expl_out = self.model.explain(image_preprocess)

        grayscale_cam = np.array(expl_out["explanation"])

        visualization_pil = Image.fromarray(np.uint8(255 * grayscale_cam))
        visualization_pil.save("test.png")

        rgb_img = np.array(image.resize((224, 224)))

        self.model.train()

        if save_expl:
            # Visualize the explanation and save it
            visualization = utils.show_cam_on_image(rgb_img / 255.0, grayscale_cam[:, :, -1])
            visualization_pil = Image.fromarray(np.uint8(visualization))
            visualization_pil.save(save_expl)

        if save_activations:
            utils.save_as_npy(grayscale_cam, save_activations)

        if return_activations:
            return grayscale_cam
        return None

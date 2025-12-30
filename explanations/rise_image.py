import numpy as np
import torch
from PIL import Image

import utils


class RiseImage:
    """Class representing Rise explanations."""

    def __init__(self, model, device=None, pasta_mode=False, lambda_pasta=0.5):
        """Initialize the Rise object.

        Parameters:
        ----------
        model: Concept Bottleneck Model to explain.
        """
        super().__init__()

        self.model = model
        self.pasta_mode = pasta_mode
        self.device = device

        if model.name == "resnet50":
            self.wrapped_model = RiseImage(
                model,
                n_masks=10000,
                initial_mask_size=(6, 6),
                p1=0.1,
                input_size=(224, 224),
                pasta_mode=pasta_mode,
                lambda_pasta=lambda_pasta,
            )  # TODO make it cleaner

        else:
            ValueError("Model not implemented !")

    def compute_and_plot_explanation(self, image, name_img=None, **kwargs):
        """Computes RISE values for a given image and saves the explanation.

        Args:
            image (PIL.Image.Image): The input image.
            num_features (int): The number of features to plot.
            type_plot (str): The type of plot to use. Can be 'waterfall', 'force' or 'bar'.
                Defaults to 'waterfall'.
            kwargs (dict): Additional keyword arguments.
                - save_expl (str): The path to save the explanation image.
            name_img (str): The name of the image (for pasta mode).
        """
        self.model.eval()
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        rgb_img = np.array(image.resize((224, 224)))

        if self.pasta_mode:
            saliency = self.wrapped_model(rgb_img, name_img)
            grayscale_cam = saliency[0]
        else:
            with torch.no_grad():
                img_tensor = self.model.preprocess(image).to(self.device).unsqueeze(0).float()
                y_numpy = self.model(img_tensor).detach().cpu().numpy()
                id_infer = np.argmax(y_numpy[0])
                saliency = self.wrapped_model(img_tensor)
            grayscale_cam = saliency[id_infer]

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

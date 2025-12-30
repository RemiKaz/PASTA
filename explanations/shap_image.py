import numpy as np
import shap
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import utils


class SHAPimage:
    """Class for computing SHAP values for a given image using a pretrained model."""

    def __init__(self, model, device, dict_hyperparam=None):
        """Initialize a SHAP image explainer.

        Args:
        model (torch.nn.Module): The pretrained model.
        device (str): The device to use for computation ('cpu' or 'cuda').
        dict_hyperparam (dict): Dictionary of hyperparameters for SHAP explainer. Defaults to None.
        """
        if dict_hyperparam is None:
            dict_hyperparam = {"masker": "blur"}
        super().__init__()

        self.device = device
        self.dict_hyperparam = dict_hyperparam
        # Initialize the SHAP image explainer
        if dict_hyperparam["masker"] == "blur":
            masker_blur = shap.maskers.Image("blur(128,128)", (224, 224, 3))
        elif dict_hyperparam["masker"] == "inpaint_telea":
            masker_blur = shap.maskers.Image("inpaint_telea")
        elif dict_hyperparam["masker"] == "inpaint_ns":
            masker_blur = shap.maskers.Image("inpaint_ns")
        self.model = model
        self.explainer = shap.Explainer(self.predict, masker_blur)

        # Fix to convet PIL image to tensor
        if self.model.name == "CLIP-zero-shot":
            self.model.preprocess = transforms.Compose(
                [transforms.ToTensor(), *model.preprocess.transforms]
            )

    def predict(self, img: np.ndarray) -> torch.Tensor:
        """Predicts the output of the model for a given image.

        Args:
            img (np.ndarray): The input image.

        Returns:
            torch.Tensor: The model's output for the input image.
        """

        def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
            """Converts a tensor from NHWC to NCHW format."""
            if x.dim() == 4:
                x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
            elif x.dim() == 3:
                x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
            return x

        img = nhwc_to_nchw(torch.Tensor(img))
        img = torch.Tensor(img)
        img = F.interpolate(img, size=(224, 224))
        img = img.to(self.device)
        return self.model(img)

    def compute_and_plot_explanation(self, image, **kwargs):
        """Computes SHAP values for a given image and saves the explanation as an image.

        Args:
            image (PIL.Image.Image): The input image.
            kwargs (dict): Additional keyword arguments.
                - model (torch.nn.Module): The pretrained model.
                - save_expl (str): The path to save the explanation image.
        """
        self.model.eval()
        model, save_expl = kwargs["model"], kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
            """Converts a tensor from NCHW to NHWC format."""
            if x.dim() == 4:
                x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
            elif x.dim() == 3:
                x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
            return x

        # Compute SHAP values
        img_tensor = self.model.preprocess(image).to(self.device).unsqueeze(0)
        out = model(img_tensor).cpu().detach().numpy()

        shap_values = self.explainer(nchw_to_nhwc(img_tensor), max_evals=200, batch_size=10)
        data = shap_values.values[0, :, :, :, np.argmax(out)].sum(-1)

        # Plot explanation
        min_val = np.min(data)
        max_val = np.max(data)
        scaled_data = data / (np.max([np.abs(min_val), np.abs(max_val)]))

        self.model.eval()

        if save_expl:
            rgb_img = np.array(image.resize((224, 224)))
            visualization = utils.show_cam_on_image(rgb_img / 255.0, scaled_data)
            visualization_pil = Image.fromarray(np.uint8(visualization))
            visualization_pil.save(save_expl)

        if save_activations:
            utils.save_as_npy(scaled_data, save_activations)

        if return_activations:
            return scaled_data
        return None

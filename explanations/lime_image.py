import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from PIL import Image

import utils


class LIMEimage:
    """Class representing a LIME image explainer."""

    def __init__(self, dict_hyperparam=None):
        """Initialize a LIME image explainer.

        This method initializes a LIME image explainer object. It creates an instance
        of the LIME image explainer class and assigns it to the `explainer` attribute.
        """
        if dict_hyperparam is None:
            dict_hyperparam = {"kernel_width": 0.25}
        super().__init__()

        # Initialize the LIME image explainer
        """self.explainer = lime_image.LimeImageExplainer(kernel_width=dict_hyperparam["kernel_lime"])"""
        self.explainer = lime_image.LimeImageExplainer()

    def compute_and_plot_explanation(self, image, **kwargs):
        """Compute and plot LIME explanations for a given image.

        Args:
            image (PIL.Image.Image): Input image.
            kwargs (dict): Additional keyword arguments.
                - model (torch.nn.Module): Model to be used for predictions.
                - save_expl (str, optional): Path to save the explanation visualization.

        Returns:
            None
        """
        model = kwargs["model"]
        model.eval()
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        save_expl = kwargs.get("save_expl", None)

        def batch_predict(images):
            """Predicts labels for a batch of images.

            Args:
                images (np.ndarray): Batch of images.

            Returns:
                np.ndarray: Batch of predicted labels.
            """
            model.eval()
            batch = torch.stack(
                tuple(model.preprocess(torch.tensor(i).float().permute(2, 0, 1)) for i in images),
                dim=0,
            )

            """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)"""

            batch = batch.to(model.device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        # Use LIME to explain the predictions
        explanation = self.explainer.explain_instance(
            np.array(image), batch_predict, top_labels=1, hide_color=0, num_samples=100
        )
        _, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=5,
            hide_rest=True,
        )
        mask = (mask + 1) * 0.49

        model.train()

        if save_expl:
            rgb_img = np.array(image.resize((224, 224)))
            visualization = utils.show_cam_on_image(rgb_img / 255.0, mask)

            # Save the explanation visualization
            visualization_pil = Image.fromarray(np.uint8(visualization))
            visualization_pil.save(save_expl)

        if save_activations:
            utils.save_as_npy(mask, save_activations)

        if return_activations:
            return mask
        return None

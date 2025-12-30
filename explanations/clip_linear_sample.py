import torch


class CLIPLinearSample:
    """Class to perform sample-based explanations using CLIP and Linear Models.

    Attributes:
        model (torch.nn.Module): CLIPLinear model for image captioning and concept disambiguation.
    """

    def __init__(self, model):
        """Initialize the CLIP_linear_sample_ class.

        Args:
            model (torch.nn.Module): CLIPLinear model for image captioning and concept disambiguation.
        """
        super().__init__()

        self.model = model

    def compute_and_plot_explanation(self, image, **kwargs):
        """Compute and plot explanations using sample-based method.

        Args:
            image (ndarray): Input image.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            save_expl (str): Path to save the explanation plot.
        """
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        self.model.eval()

        with torch.no_grad():
            scores = self.model.scores_from_images(image)

            # Compute and plot counterfactual explanations using Linear Models
            activations = self.model.plot_linear_explanation(
                scores,
                top_concepts=5,
                custom_fig_name=save_expl,
                save_activations=save_activations,
                return_activations=return_activations,
            )

        self.model.train()

        if return_activations:
            return activations
        return None

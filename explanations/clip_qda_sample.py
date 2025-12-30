import torch


class CLIPQDAsample:
    """Class to perform sample-based explanations using CLIP and Gaussian Mixture Models.

    Attributes:
        model: CLIPQDA model for image captioning and concept disambiguation.
    """

    def __init__(self, model, device):
        """Initialize the CLIPQDAsample class.

        Args:
            model: CLIPQDA model for image captioning and concept disambiguation.
            device: Device to use for computation.
        """
        super().__init__()

        self.model = model
        self.device = device

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

        if self.model.pytorch_mode:
            self.model.to(self.device)

        with torch.no_grad():
            # Compute scores from the input image
            scores = self.model.scores_from_images(image)

            # Compute and plot counterfactual explanations using Gaussian Mixture Models
            activations = self.model.compute_and_plot_conterfactual_explaination_gmm(
                scores,  # Scores from the input image
                top_concepts=5,  # Number of top concepts to consider
                scale_var=True,  # Whether to scale the variance of the GMM
                custom_fig_name=save_expl,  # Path to save the explanation plot
                save_activations=save_activations,
                return_activations=return_activations,
            )

        self.model.train()

        if return_activations:
            return activations
        return None

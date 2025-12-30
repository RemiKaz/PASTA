class XnesylLinearSample:
    """Class to perform sample-based explanations using Xnesyl and Linear Models.

    Attributes:
        model (torch.nn.Module): Xnesyl model for image captioning and concept disambiguation.
    """

    def __init__(self, model, device=None):
        """Initialize the CLIP_linear_sample_ class.

        Args:
            model (torch.nn.Module): Xnesyl model for image captioning and concept disambiguation.
            device (torch.device, optional): Device to use for computation. If None, use the default device.
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
        self.model.eval()
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        if self.model.pytorch_mode:
            self.model.backbone.eval()

            # Compute scores from the input image
            scores = self.model.scores_from_images(image)

            self.model.to(self.device)

            # Compute and plot counterfactual explanations using Linear Models
            activations = self.model.plot_linear_explanation(
                scores,
                top_concepts=5,
                custom_fig_name=save_expl,
                save_activations=save_activations,
                return_activations=return_activations,
            )

            self.model.backbone.train()

        else:
            self.model.backbone.eval()

            # Compute scores from the input image
            scores = self.model.scores_from_images(image)

            # Compute and plot counterfactual explanations using Linear Models
            activations = self.model.plot_linear_explanation(
                scores,
                top_concepts=5,
                custom_fig_name=save_expl,
                save_activations=save_activations,
                return_activations=return_activations,
            )

            self.model.backbone.train()

        return activations

import numpy as np
import torchvision.transforms as transforms
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

        elif model.name == "vitB":

            def reshape_transform(tensor, height=14, width=14):
                """Reshape tensor from (batch_size, num_channels, height, width) to
                (batch_size, height, width, num_channels).
                """
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                # Bring the channels to the first dimension,
                # like in CNNs.
                return result.permute(0, 3, 1, 2).float()

            target_layers = [model.vit_b_16.encoder.layers.encoder_layer_11.ln_1]

            self.explainer = ScoreCAM_(
                model=model,
                target_layers=target_layers,
                reshape_transform=reshape_transform,
            )

        elif model.name == "CLIP-zero-shot":

            def reshape_transform(tensor, height=7, width=7):
                """Reshape tensor from (batch_size, num_channels, height, width) to
                (batch_size, height, width, num_channels).
                """
                tensor = tensor.permute(1, 0, 2)
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                # Bring the channels to the first dimension,
                # like in CNNs.
                return result.permute(0, 3, 1, 2).float()

            self.model.preprocess = transforms.Compose(
                [transforms.ToTensor(), *self.model.preprocess.transforms]
            )

            def reshape_transform_openclip(tensor, image_size=224):
                """Reshape tensor from (batch_size, num_channels, height, width) to
                (batch_size, height, width, num_channels).
                """
                height = image_size // model.clip_net.visual.patch_size[0]
                width = image_size // model.clip_net.visual.patch_size[1]
                """tensor = tensor.permute(1, 0, 2)"""
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                # Bring the channels to the first dimension,
                # like in CNNs.
                return result.permute(0, 3, 1, 2).float()

            target_layers = [model.clip_net.visual.transformer.resblocks[-2].ln_1]

            if model.is_open_clip:
                # For OpenCLIP, we need to use the reshape_transform
                self.explainer = ScoreCAM_(
                    model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform_openclip,
                )

            else:
                self.explainer = ScoreCAM_(
                    model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform,
                )
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
        self.model.eval()
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        img_tensor = self.model.preprocess(image).to(self.device).unsqueeze(0).float()

        # Compute ScoreCAM explanation
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

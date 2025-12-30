import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor

import utils


class VitB(nn.Module):
    """This class is a PyTorch Module that implements a Vision Transformer (ViT)
    backbone followed by a linear layer for image classification.
    """

    def __init__(self, output_dim, device, freeze_backbone=False, bcos=False, bcos_eval=False):
        """Initializes the ViT backbone and the linear layer for image classification.

        Parameters:
                output_dim (int): The number of classes in the output.
                device (str): The device to run the model on, either 'cpu' or 'cuda'.
        freeze_backbone (bool): If True, the backbone weights won't be updated during training.
        """
        super().__init__()

        # Import model
        self.freeze_backbone = freeze_backbone
        self.bcos = bcos

        if bcos:
            self.name = "vitB-bcos"

            # load a pretrained model

            if bcos_eval:
                vit_b_16_all = (
                    torch.hub.load("B-cos/B-cos-v2", "simple_vit_b_patch16_224", pretrained=True)
                    .eval()
                    .to(device)
                )
                # Load locally
                """vit_b_16_all = torch.load(
                    "models_pkl/bcos_simple_vit_b_patch16_224-1fc4750806.pth",
                    map_location=device).eval()"""

                self.explain = vit_b_16_all.explain

            else:
                vit_b_16_all = torch.hub.load(
                    "B-cos/B-cos-v2", "simple_vit_b_patch16_224", pretrained=True
                )

            self.preprocess = vit_b_16_all.transform
            self.add_inverse = utils.AddInverse()

            # Replace the lest linear layer
            self.vit_b_16 = torch.nn.Sequential(*(list(vit_b_16_all.children())[:-1])).to(device)
            self.vit_b_16[0].linear_head.linear.linear = NormedLinear(
                in_features=768, out_features=768, bias=False
            ).to(device)
            self.device = device
            self.linear = nn.Linear(768, output_dim).to(device)

        else:
            self.name = "vitB"
            vit_b_16_all = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
            self.preprocess = ViT_B_16_Weights.DEFAULT.transforms(antialias=True)
            self.device = device
            # Replace the lest linear layer
            self.vit_b_16 = create_feature_extractor(vit_b_16_all, return_nodes=["getitem_5"])
            self.linear = nn.Linear(768, output_dim).to(
                device
            )  # The linear layer for classification.

    def forward(self, x, pred_class_mode=False):
        """Forward pass of the model.

        Parameters:
                x (torch.Tensor): The input tensor.

        Returns:
                torch.Tensor: The output tensor.
        """
        if self.bcos:
            if not pred_class_mode:
                x = self.add_inverse(x)
            if self.freeze_backbone:
                with torch.no_grad():
                    x = self.vit_b_16(x)
            else:
                x = self.vit_b_16(x)

            return self.linear(x)

        if self.freeze_backbone:
            with torch.no_grad():
                x = self.vit_b_16(x)["getitem_5"]
        else:
            x = self.vit_b_16(x)["getitem_5"]

        return self.linear(x)

    def predict_class(self, x, eval_mode=False):
        """Given an input tensor, predicts the class.

        Parameters:
                x (PIL.Image): The input tensor.

        Returns:
                torch.Tensor: The predicted class.
        """
        if eval_mode:
            x = self.preprocess(x).to(self.device).unsqueeze(0)
            self.eval()
            with torch.no_grad():
                x = self.forward(x, pred_class_mode=True)
            self.train()
            return torch.argmax(x, dim=1)

        x = self.preprocess(x).to(self.device).unsqueeze(0)
        x = self.forward(x, pred_class_mode=True)
        return torch.argmax(x, dim=1)


class NormedLinear(nn.Linear):
    """Standard linear transform, but with unit norm weights."""

    def forward(self, input_tensor: Tensor) -> Tensor:
        w = self.weight / LA.vector_norm(self.weight, dim=1, keepdim=True)
        return F.linear(input_tensor, w, self.bias)

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

import utils


class Resnet50(nn.Module):
    """ResNet50 model with a linear layer on top.

    Args:
        output_dim (int): Size of the output layer.
        device (str): Device to run the model on.
        freeze_backbone (bool, optional): Freeze the backbone layers.
            Defaults to False.
    """

    def __init__(self, output_dim, device, freeze_backbone=False, bcos=False, bcos_eval=False):
        super().__init__()

        # Import model
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.bcos = bcos

        if bcos:
            self.name = "resnet50-bcos"

            # load a pretrained model

            if bcos_eval:
                resnet50_all = (
                    torch.hub.load("B-cos/B-cos-v2", "resnet50", pretrained=True).eval().to(device)
                )
                """resnet50_all = torch.load(
                    "models_pkl/resnet_50-ead259efe4.pth",
                    map_location=device).eval()"""
                self.explain = resnet50_all.explain

            else:
                resnet50_all = torch.hub.load("B-cos/B-cos-v2", "resnet50", pretrained=True)
                """resnet50_all = torch.load(
                    "models_pkl/resnet_50-ead259efe4.pth",
                    map_location=device)"""

            self.preprocess = resnet50_all.transform
            self.add_inverse = utils.AddInverse()

            # Replace the lest linear layer
            self.resnet50 = torch.nn.Sequential(*(list(resnet50_all.children())[:-2])).to(device)
            self.linear = nn.Linear(2048, output_dim).to(device)

        else:
            self.name = "resnet50"

            # load a pretrained model
            resnet50_all = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
            self.preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=True)

            # Replace the lest linear layer
            self.resnet50 = torch.nn.Sequential(*(list(resnet50_all.children())[:-1]))
            self.linear = nn.Linear(2048, output_dim).to(device)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.bcos:
            x = self.add_inverse(x)

        if self.freeze_backbone:
            with torch.no_grad():
                x = self.resnet50(x)
        else:
            x = self.resnet50(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.linear(x)

    def predict_class(self, x, eval_mode=False):
        """Predicts the class of the input image.

        Args:
            x (PIL.Image): Input image.
            eval_mode (bool): Whether to run in eval mode.

        Returns:
            torch.Tensor: Predicted class.
        """
        flag_bcos = True
        if self.bcos:  # Remove add_inverse in forward because it is in preprocess
            self.bcos = False
            flag_bcos = True

        if eval_mode:
            x = self.preprocess(x).to(self.device).unsqueeze(0)
            self.eval()
            with torch.no_grad():
                x = self.forward(x)
            self.train()
            return torch.argmax(x, dim=1)

        x = self.preprocess(x).to(self.device).unsqueeze(0)
        x = self.forward(x)

        if flag_bcos:
            self.bcos = True

        return torch.argmax(x, dim=1)

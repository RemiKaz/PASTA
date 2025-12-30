import clip
import open_clip
import torch
from torch import nn
from torchvision import transforms


class CLIPzeroshot(nn.Module):
    """Implements the CLIP-zero-shot model for text-image matching."""

    def __init__(self, list_labels, device, backbone="ViT-B/32", pretrained=None):
        super().__init__()
        """
        Initialize the CLIP-zero-shot model.
        Params:
            list_labels (list): List of labels.
            device (str): Device to run the model on.
            backbone (str): Model backbone name.
            pretrained (str, optional): Pretrained weights for OpenCLIP models.
        """
        self.name = "CLIP-zero-shot"
        self.device = device

        if pretrained is not None:
            # Load OpenCLIP model
            if backbone == "ViT-g-14":
                model, _, preprocess = open_clip.create_model_and_transforms(
                    backbone,
                    pretrained="/lustre/fswork/projects/rech/wqn/ufb58bn/ProjetsRemi/Dataset_XAI/models_pkl/open_clip_pytorch_model.bin",
                    device=self.device,
                )
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    backbone, pretrained=pretrained, device=self.device
                )
            self.clip_net = model
            self.preprocess_clip = preprocess
            tokenizer = open_clip.get_tokenizer(backbone)
            self.is_open_clip = True
        else:
            # Load original CLIP model
            self.clip_net, self.preprocess_clip = clip.load(backbone, device=self.device)
            tokenizer = clip.tokenize
            self.is_open_clip = False

        def get_preprocess_transform():
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            return transforms.Compose(
                [
                    transforms.Resize(
                        224,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    normalize,
                ]
            )

        self.preprocess = get_preprocess_transform()

        self.clip_net.eval().to(torch.float32)

        # Tokenize the concepts/labels and compute the text embeddings
        self.list_labels = list_labels
        tokens = tokenizer(list_labels).to(self.device)
        with torch.no_grad():
            self.text_embeding = self.clip_net.encode_text(tokens)
        # Normalize the text embeddings once during initialization
        self.text_embeding = self.text_embeding / self.text_embeding.norm(dim=-1, keepdim=True)

    def forward(self, x):
        """Compute the similarity between the input image and the concepts/labels.

        Args:
            x (PIL.Image.Image or numpy.ndarray): Input image.

        Returns:
            torch.Tensor: Similarity scores.
        """
        # Compute scores for the input image
        img_tensor = x.to(torch.float32)
        image_embedding = self.clip_net.encode_image(img_tensor)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        text_features = self.text_embeding  # already normalized
        return (100.0 * image_embedding @ text_features.T).softmax(dim=-1)

    def predict_class(self, x, eval_mode=False):
        """Predict the class of the input image.

        Args:
            x (PIL.Image.Image or numpy.ndarray): Input image.
            eval_mode (bool, optional): If True, the model is in evaluation mode. Defaults to False.

        Returns:
            torch.Tensor: Predicted class.
        """
        # Compute the similarity and predict the class
        if eval_mode:
            x = self.preprocess_clip(x).unsqueeze(0).to(self.device)
            self.eval()
            with torch.no_grad():
                x = self.forward(x)
            self.train()
            return torch.argmax(x, dim=1)

        x = self.preprocess_clip(x).unsqueeze(0).to(self.device)
        x = self.forward(x)
        return torch.argmax(x, dim=1)

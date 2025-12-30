import clip
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn

import utils


class CLIPLinear(nn.Module):
    def __init__(self, list_concepts, list_classes, device):
        super().__init__()
        """
		Initializes a CLIP_Linear instance.

		Args:
			list_concepts (list): List of concepts.
			list_classes (list): List of classes.
			device (str): Device to run on.
		"""

        self.name = "CLIP-Linear"
        self.device = device

        # Load CLIP model
        self.clip_net, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_net.eval()

        # Store concepts and classes
        self.list_concepts = list_concepts
        tokens = clip.tokenize(list_concepts).to(self.device)
        self.text_embeding = self.clip_net.encode_text(tokens).float()
        self.list_classes = list_classes

        # Linear parameters
        self.linear = nn.Linear(len(list_concepts), len(list_classes)).to(self.device)

        # Custom colormap for matplotlib
        colors2 = plt.cm.coolwarm_r(np.linspace(0.5, 1, 128))
        colors1 = plt.cm.coolwarm_r(np.linspace(0, 0.5, 128))
        colors = np.vstack((colors2, colors1))
        mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
        mpl.cm.register_cmap("mycolormap", mymap)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        with torch.no_grad():
            clip_embeds = torch.tensor(x).to(self.device).float()
            clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)
            text_features = self.text_embeding / self.text_embeding.norm(dim=-1, keepdim=True)
            similarity = 100.0 * clip_embeds @ text_features.T
        return self.linear(similarity)

    def preprocess_clip_scores(self, clip_embeds, verbose=True, save_train_score=False):
        """Computes the CLIP scores for the given image embeddings.

        Parameters:
        clip_embeds (numpy.ndarray): Array of image embeddings.
        verbose (bool, optional): Flag indicating whether to print a progress message. Defaults to True.
        save_train_score (bool, optional): Flag indicating whether to save the computed CLIP scores. Defaults to False.

        Returns:
        numpy.ndarray: Array of CLIP scores.
        """
        if verbose:
            print("Computing CLIP scores...")

        clip_scores = np.empty((0, len(self.list_concepts)))

        # Compute CLIP scores for each image embedding
        clip_embeds = torch.tensor(clip_embeds).to(self.device)
        for embeds in clip_embeds:
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            text_features = self.text_embeding / self.text_embeding.norm(dim=-1, keepdim=True)
            similarity = (100.0 * embeds @ text_features.T).unsqueeze(0).cpu().detach().numpy()
            clip_scores = np.concatenate((clip_scores, similarity), axis=0)

        # Save score if needed
        if save_train_score:
            self.clip_scores_train = clip_scores

        return clip_scores

    def test(self, data_in, data_out):
        """Computes the accuracy score of the trained linear model on the provided data.

        Parameters:
        data_in (numpy.ndarray): Array of image embeddings.
        data_out (numpy.ndarray): Array of output labels.

        Returns:
        float: The accuracy score of the model.
        """
        # Preprocess CLIP scores for testing
        clip_scores = self.preprocess_clip_scores(data_in)
        n_sucess = 0
        for score, label in zip(clip_scores, data_out, strict=False):
            pred = self.infer_from_scores(score)
            n_sucess += sum(pred == label)
        return n_sucess / len(data_out)

    def infer_from_images(self, x):
        """Given an input image, predicts the label using the trained model.

        Parameters:
        input (PIL.Image.Image or np.ndarray): Input image.

        Returns:
        int: Predicted label.
        """
        # Compute scores for the input image
        img_tensor = self.preprocess(x).unsqueeze(0).to(self.device)
        image_embedding = self.clip_net.encode_image(img_tensor)
        image_features = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        text_features = self.text_embeding / self.text_embeding.norm(dim=-1, keepdim=True)
        similarity = 100.0 * image_features.float() @ text_features.T

        # Predict the label using the trained model
        return torch.argmax(self.linear(similarity))

    def scores_from_images(self, x):
        """Computes the similarity scores between an input image and the text embeddings.

        Parameters:
        input (PIL.Image or np.ndarray): Input image.

        Returns:
        numpy.ndarray: Array of similarity scores.
        """
        # Compute scores for the input image
        img_tensor = self.preprocess(x).unsqueeze(0).to(self.device)
        image_embedding = self.clip_net.encode_image(img_tensor)
        image_features = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        image_features = image_features[0]
        text_features = self.text_embeding / self.text_embeding.norm(dim=-1, keepdim=True)
        return (100.0 * image_features.float() @ text_features.T).cpu().detach().numpy()

    def infer_from_scores(self, input_score):
        """Given input similarity scores, predicts the label using the trained model.

        Parameters:
        input_score (numpy.ndarray): Array of similarity scores.

        Returns:
        int: Predicted label.
        """
        input_score = torch.tensor(input_score.reshape(1, -1)).to(self.device)

        # Predict the label using the trained model
        return torch.argmax(self.linear(input_score))

    def infer_proba_from_scores(self, input_score):
        """Given input similarity scores, predicts the label probabilities using the trained model.

        Parameters:
        input_score (numpy.ndarray): Array of similarity scores.

        Returns:
        numpy.ndarray: Array of label probabilities.
        """
        input_score = input_score.reshape(1, -1)

        # Predict the label probabilities using the trained model
        return F.softmax(self.linear(input_score))

    def predict_class(self, x, eval_mode=False):
        """Predicts the class of the input images. (Added to homogenize the function predict_class with the other models).

        Parameters:
        x (torch.Tensor): Input images.

        Returns:
        int: Predicted class.
        """
        # Infer labels from input images
        if eval_mode:
            self.eval()
            with torch.no_grad():
                output = self.infer_from_images(x).unsqueeze(0)
            self.train()
            return output

        return self.infer_from_images(x).unsqueeze(0)

    def plot_linear_explanation(
        self,
        scores_image,
        top_concepts=5,
        custom_fig_name=False,
        save_activations=False,
        return_activations=False,
    ):
        """Plots the linear explanation for the input image_scores.

        Parameters:
        scores_image (numpy.ndarray): Array of similarity scores.
        top_concepts (int, optional): Number of concepts to plot. Defaults to 5.
        custom_fig_name (bool, optional): Whether to use a custom figure name. Defaults to False.
        """
        scores_image = scores_image[np.newaxis, ...]
        label_id = (
            self.infer_from_scores(scores_image).cpu().detach().numpy()
        )  # Infer label from input data
        inputs = scores_image[0]
        a_heatmap_classes = self.linear.weight.cpu().detach().numpy()  # N_classes , N_concepts

        list_influence_attributes = np.zeros(len(self.list_concepts))

        for i_attribute in range(len(self.list_concepts)):
            list_influence_attributes[i_attribute] = (
                a_heatmap_classes[label_id][i_attribute] * inputs[i_attribute]
            )

        list_influence_attributes_abs = np.abs(list_influence_attributes)

        list_influence_attributes_abs, list_influence_attributes, list_best_attributes = zip(
            *sorted(
                zip(
                    list_influence_attributes_abs,
                    list_influence_attributes,
                    self.list_concepts.copy(),
                    strict=False,
                ),
                reverse=True,
            ),
            strict=False,
        )

        if save_activations:
            d_data_in_full = {
                list_best_attributes[i]: list_influence_attributes[i]
                for i in range(len(list_best_attributes))
            }
            utils.save_as_json(d_data_in_full, save_activations)

        if return_activations:
            return {
                list_best_attributes[i]: list_influence_attributes[i]
                for i in range(len(list_best_attributes))
            }

        a_distances_ordered = list_influence_attributes[:top_concepts]
        labels = list_best_attributes[:top_concepts]

        d_data_in = {labels[i]: a_distances_ordered[i] for i in range(len(labels))}
        pd_data_in = pd.DataFrame(d_data_in, index=[0])
        sns.set_theme(font_scale=1)
        sns.barplot(
            pd_data_in,
            palette=utils.colors_from_values(a_distances_ordered, "coolwarm"),
            orient="h",
        )

        if custom_fig_name:
            """plt.bar(L_attributes_ordered[:top_concepts],a_distances_ordered[:top_concepts],width=0.4)"""
            plt.xlabel("Weight value", fontsize=20, fontweight="bold")
            plt.ylabel("Concept", fontsize=20, fontweight="bold")
            plt.tight_layout()
            plt.savefig(custom_fig_name)
            return None
        return None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn

import utils


class Xnesyl(nn.Module):
    def __init__(
        self,
        list_concepts,
        list_classes,
        device,
        backbone_name="faster-rcnn",
        classifier_name="logistic",
        pytorch_mode=False,
    ):
        """Initializes a Xnesyl instance that predict concepts from images (backbone) and labels from concepts (classifier).
        Version without SHAP backpropagations
        https://arxiv.org/pdf/2104.11914v2.

        Args:
            list_concepts (list): List of concepts.
            list_classes (list): List of classes.
            device (str): Device to run on.
            backbone_name (str): Name of the backbone model.
            classifier_name (str): Name of the classifier model.
            pytorch_mode (bool): A boolean indicating whether to use PyTorch to build the classifier.
        """
        super().__init__()
        # Initialize the list of concepts, classes, and device
        self.pytorch_mode = pytorch_mode
        self.list_concepts = list_concepts
        self.list_classes = list_classes
        self.device = device
        self.preprocess_base = ResNet50_Weights.DEFAULT.transforms(antialias=True)

        # Initialize the backbone model based on the provided backbone name
        if backbone_name == "faster-rcnn":
            self.backbone = fasterrcnn_resnet50_fpn(pretrained=True, device=device).to(device)
            self.backbone.name = "XNES-backbone-faster-rcnn"
            in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
            self.backbone.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
                in_features, len(list_concepts)
            ).to(device)

        else:
            raise ValueError("Backbone not implemented !")

        # Initialize the classifier model based on the provided classifier name
        if classifier_name == "linear":
            self.classifier = LinearRegression()
            self.classifier.name = "XNES-classifier-linear"
            self.name = "XNES-classifier-linear"

        elif classifier_name == "logistic":
            self.classifier = LogisticRegression()
            self.classifier.name = "XNES-classifier-logistic"
            self.name = "XNES-classifier-logistic"
            if self.pytorch_mode:
                self.torch_classifier = nn.Sequential(
                    nn.Linear(len(self.list_concepts), len(self.list_classes)), nn.Sigmoid()
                )
        else:
            raise ValueError("Classifier not implemented !")

    def train_classifier(self, data_in, data_out, save_score=True):
        """Trains the classifier model with the provided input data and output labels.

        Args:
            data_in (numpy.ndarray): Input data for training.
            data_out (numpy.ndarray): Output labels for training.
            save_score (bool): A boolean indicating whether to save the training scores.
        """
        # Fit the classifier model
        self.classifier.fit(data_in, data_out)

        if self.pytorch_mode:
            with torch.no_grad():
                # Set weights
                self.torch_classifier[0].weight.copy_(
                    torch.tensor(self.classifier.coef_, dtype=torch.float32)
                )

                # Set bias
                self.torch_classifier[0].bias.copy_(
                    torch.tensor(self.classifier.intercept_, dtype=torch.float32)
                )

        if save_score:
            self.scores_train = data_in

    def test(self, data_in, data_out):
        """Compute the accuracy score of the trained classifier.

        Args:
            data_in (numpy.ndarray): Input data for testing.
            data_out (numpy.ndarray): Output labels for testing.

        Returns:
            float: The accuracy score of the model.
        """
        inputs = torch.tensor(data_in, dtype=torch.float32).to(self.device)
        labels = torch.tensor(data_out, dtype=torch.long).to(
            self.device
        )  # Assuming classification (long tensor for labels)

        if self.pytorch_mode:
            self.torch_classifier.to(self.device)
            self.torch_classifier.eval()  # Set model to evaluation mode

            correct = 0
            total = 0

            with torch.no_grad():  # No need to compute gradients during evaluation
                # Forward pass: compute predictions
                outputs = self.torch_classifier(inputs)

                # Get predicted classes (for classification tasks)
                _, predicted = torch.max(outputs.data, 1)

                # Count correct predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Compute mean accuracy
            return correct / total

        return self.classifier.score(data_in, data_out)

    def scores_from_images(self, images):
        """Preprocesses the images and computes the scores using the backbone model.

        Args:
            images (PIL.Image or np.ndarray): Images to preprocess and compute scores.

        Returns:
            numpy.ndarray: Array of computed scores.
        """
        images = (
            self.preprocess_base(images).unsqueeze(0).to(self.device)
        )  # Bad because I normalize the image two times TODO training with correct normalization

        outputs = self.backbone(images)

        # Feature importance: highest probability among the detected bbox
        pred_labels = outputs[0]["labels"].detach().cpu().numpy()
        scores = outputs[0]["scores"].detach().cpu().numpy()

        scores_cbm = np.empty(len(self.list_concepts))

        for j, _ in enumerate(self.list_concepts):
            if j in pred_labels:
                # Print the highest score that correspond to the concept
                scores_cbm[j] = max([scores[k] for k in range(len(scores)) if pred_labels[k] == j])
            else:
                scores_cbm[j] = 0

        return scores_cbm

    def forward(self, x):
        outputs = self.backbone(x)

        # Feature importance: highest probability among the detected bbox
        pred_labels = outputs[0]["labels"].detach().cpu().numpy()
        scores = outputs[0]["scores"].detach().cpu().numpy()

        scores_cbm = np.empty(len(self.list_concepts))

        for j, _ in enumerate(self.list_concepts):
            if j in pred_labels:
                # Print the highest score that correspond to the concept
                scores_cbm[j] = max([scores[k] for k in range(len(scores)) if pred_labels[k] == j])
            else:
                scores_cbm[j] = 0

        if self.pytorch_mode:
            scores_cbm_tensor = torch.tensor(scores_cbm, dtype=torch.float32).to(self.device)
            output = self.torch_classifier(scores_cbm_tensor)

        else:
            output = self.infer_from_scores(scores_cbm)

        return output

    def infer_from_scores(self, input_score):
        """Given input similarity scores, predicts the label using the trained model.

        Parameters:
        input_score (numpy.ndarray): Array of similarity scores.

        Returns:
        int: Predicted label.
        """
        input_score = input_score.reshape(1, -1)

        if self.pytorch_mode:
            input_score_tensor = torch.tensor(input_score, dtype=torch.float32).to(self.device)
            return (
                torch.argmax(self.torch_classifier(input_score_tensor))
                .unsqueeze(0)
                .detach()
                .cpu()
                .numpy()
            )

        # Predict the label using the trained model
        return self.classifier.predict(input_score)

    def infer_from_images(self, image):
        """Given an input image, predicts the label using the trained model.

        Parameters:
        input (PIL.Image.Image or np.ndarray): Input image.

        Returns:
        int: Predicted label.
        """
        # Compute scores for the input image
        scores = self.scores_from_images(image)

        # Compute label from score
        return self.infer_from_scores(scores)

    def predict_class(self, x, eval_mode=False):
        """Predicts the class of the input images. (Added to homogenize the function predict_class with the other models).

        Parameters:
        x (torch.Tensor): Input images.

        Returns:
        int: Predicted class.
        """
        # Infer labels from input images
        if eval_mode:
            if self.pytorch_mode:
                self.eval()
                output = self.infer_from_images(x)
                self.train()
            else:
                self.backbone.eval()
                output = self.infer_from_images(x)
                self.backbone.train()
            return output

        return self.infer_from_images(x)

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

        label_id = self.infer_from_scores(scores_image)[0]  # Infer label from input data

        inputs = scores_image[0]

        if self.pytorch_mode:
            a_heatmap_classes = self.torch_classifier[0].weight.detach().cpu().numpy()
        else:
            a_heatmap_classes = self.classifier.coef_  # N_classes , N_concepts

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

        if all(x == 0 for x in a_distances_ordered):
            sns.barplot(
                pd_data_in,
                orient="h",
            )

        else:
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

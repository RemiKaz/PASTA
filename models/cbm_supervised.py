import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from torch import nn

import models


class CBMsupervised(nn.Module):
    def __init__(
        self,
        list_concepts,
        list_classes,
        device,
        backbone_name="resnet",
        classifier_name="linear",
        pytorch_mode=False,
    ):
        """Initializes a Concept Bottleneck Model instance that predict concepts from images (backbone) and labels from concepts (classifier).

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
        self.list_concepts = list_concepts
        self.list_classes = list_classes
        self.device = device
        self.pytorch_mode = pytorch_mode

        # Initialize the backbone model based on the provided backbone name
        if backbone_name == "resnet":
            self.backbone = models.Resnet50(output_dim=len(list_concepts), device=device)
            self.backbone.name = "CBM-backbone-resnet"
        elif backbone_name == "vitB":
            self.backbone = models.VitB(output_dim=len(list_concepts), device=device)
        else:
            raise ValueError("Backbone not implemented !")

        # Initialize the classifier model based on the provided classifier name
        if classifier_name == "linear":
            self.classifier = LinearRegression()
            self.classifier.name = "CBM-classifier-linear"
            self.name = "CBM-classifier-linear"

        elif classifier_name == "logistic":
            self.classifier = LogisticRegression()
            self.classifier.name = "CBM-classifier-logistic"
            self.name = "CBM-classifier-logistic"
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

        self.mean_train = np.mean(data_in).astype(np.float32)  # For occlusion/rise

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
            mean_accuracy = correct / total

            print("pytorch", mean_accuracy)

            return mean_accuracy

        return self.classifier.score(data_in, data_out)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.backbone(x)
        if self.pytorch_mode:
            return self.torch_classifier(x)

        return self.classifier.predict_proba(x.cpu().detach().numpy())

    def scores_from_images(self, images):
        """Preprocesses the images and computes the scores using the backbone model.

        Args:
            images (PIL.Image or np.ndarray): Images to preprocess and compute scores.

        Returns:
            numpy.ndarray: Array of computed scores.
        """
        images = self.backbone.preprocess(images).to(self.device).unsqueeze(0)
        return self.backbone(images).squeeze(0).detach().cpu().numpy()

    def infer_proba_from_scores(self, input_score):
        """Given input similarity scores, predicts the label using the trained model.

        Parameters:
        input_score (numpy.ndarray): Array of similarity scores.

        Returns:
        int: Predicted label.
        """
        if len(input_score.shape) == 1:
            input_score = input_score.reshape(1, -1)

        if self.pytorch_mode:
            input_score_tensor = torch.tensor(input_score, dtype=torch.float32).to(self.device)
            return self.torch_classifier(input_score_tensor).detach().cpu().numpy()

        # Predict the label using the trained model
        return self.classifier.predict_proba(input_score)

    def infer_from_scores(self, input_score):
        """Given input similarity scores, predicts the label using the trained model.

        Parameters:
        input_score (numpy.ndarray): Array of similarity scores.

        Returns:
        int: Predicted label.
        """
        if len(input_score.shape) == 1:
            input_score = input_score.reshape(1, -1)

        if self.pytorch_mode:
            input_score_tensor = torch.tensor(input_score, dtype=torch.float32).to(self.device)
            """print('input_score_tensor',input_score_tensor.size())
            print('torch_classifier',self.torch_classifier[0].weight.size())"""
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
            self.backbone.eval()
            if self.pytorch_mode:
                self.torch_classifier.eval()
            with torch.no_grad():
                output = self.infer_from_images(x)
            self.backbone.train()
            return output

        return self.infer_from_images(x)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lime.lime_tabular import LimeTabularExplainer as LimeTabularExplainer_origin

import utils


class LIMECBM:
    """Class to compute and plot LIME explanations for CBM models."""

    def __init__(self, model):
        """Constructor of the LIMECBM class.

        Args:
            model (CBM): A CBM model instance.
        """
        super().__init__()

        self.model = model

        # Initialize Lime for Tabular data

        if model.name == "CLIP-QDA":
            self.explainer = LimeTabularExplainer_origin(
                self.model.clip_scores_train,  # CLIP scores from training data
                feature_names=self.model.list_concepts,  # List of concepts
                class_names=self.model.list_classes,  # List of classes
                discretize_continuous=True,  # Discretize continuous flag
            )

        elif model.name == "CBM-classifier-logistic":
            self.model.backbone.eval()
            self.explainer = LimeTabularExplainer_origin(
                self.model.scores_train,  # Scores from training data
                feature_names=self.model.list_concepts,  # List of concepts
                class_names=self.model.list_classes,  # List of classes
                discretize_continuous=True,  # Discretize continuous flag
            )

        else:
            raise ValueError("Model not implemented !")

    def compute_and_plot_explanation(self, image, num_features=5, **kwargs):
        """Compute and plot LIME explanations for a given image.

        Args:
            image (numpy.ndarray): Input image.
            num_features (int, optional): Number of features to consider.
                Defaults to 5.
            kwargs (dict): Additional keyword arguments.

        Returns:
            None
        """
        self.model.eval()
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        # Get the scores for the input image from the model
        scores = self.model.scores_from_images(image)

        # Compute LIME explanation for the input image
        d_data_in = {
            self.model.list_concepts[i]: scores[np.newaxis, ...][:, i]
            for i in range(len(self.model.list_concepts))
        }

        pd_data_in = pd.DataFrame(d_data_in)

        if self.model.name == "CLIP-QDA":
            if self.model.pytorch_mode:
                self.model.eval()
                self.model.to(self.model.device)
                exp = self.explainer.explain_instance(
                    scores,
                    self.model.infer_proba_from_scores,
                    num_features=len(self.model.list_concepts),
                    top_labels=1,
                )
            else:
                exp = self.explainer.explain_instance(
                    scores,
                    self.model.QDA.predict_proba,
                    num_features=len(self.model.list_concepts),
                    top_labels=1,
                )

        elif self.model.name == "CBM-classifier-logistic":
            if self.model.pytorch_mode:
                self.model.eval()
                self.model.to(self.model.device)
                exp = self.explainer.explain_instance(
                    scores,
                    self.model.infer_proba_from_scores,
                    num_features=len(self.model.list_concepts),
                    top_labels=1,
                )
            else:
                exp = self.explainer.explain_instance(
                    scores,
                    self.model.classifier.predict_proba,
                    num_features=len(self.model.list_concepts),
                    top_labels=1,
                )

        # Infer the prediction from the model
        scores = torch.from_numpy(scores).unsqueeze(0)
        prediction = int(self.model.infer_from_scores(scores))
        print(prediction)

        self.model.train()
        if save_activations:
            exp_all = exp.as_list(label=prediction)
            vals_all = [x[1] for x in exp_all]
            names_all = [x[0] for x in exp_all]
            d_data_in_full = {names_all[i]: vals_all[i] for i in range(len(vals_all))}
            utils.save_as_json(d_data_in_full, save_activations)

        if return_activations:
            exp_all = exp.as_list(label=prediction)
            vals_all = [x[1] for x in exp_all]
            names_all = [x[0] for x in exp_all]
            return {names_all[i]: vals_all[i] for i in range(len(vals_all))}

        # Plot the explanation
        exp = exp.as_list(label=prediction)[:num_features]

        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        d_data_in = {names[i]: vals[i] for i in range(len(vals))}
        pd_data_in = pd.DataFrame(d_data_in, index=[0])
        sns.set_theme(font_scale=1)
        sns.barplot(pd_data_in, palette=utils.colors_from_values(vals, "coolwarm"), orient="h")
        plt.xlabel("Weight value", fontsize=20, fontweight="bold")
        plt.ylabel("Concept", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_expl)
        return None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import utils


class SHAPCBM:
    """Class representing SHAP CBM explanations."""

    def __init__(self, model):
        """Initialize the SHAP CBM object.

        Parameters:
        ----------
        model: Concept Bottleneck Model to explain.
        """
        super().__init__()

        self.model = model

        # Initialize SHAP for Tabular data
        if model.name == "CLIP-QDA":
            if model.pytorch_mode:
                x_train_summary = shap.kmeans(self.model.clip_scores_train, 5)
                self.explainer = shap.KernelExplainer(
                    self.model.infer_proba_from_scores, x_train_summary
                )
            else:
                self.model.to(self.model.device)
                x_train_summary = shap.kmeans(self.model.clip_scores_train, 5)
                self.explainer = shap.KernelExplainer(self.model.QDA.predict_proba, x_train_summary)

        elif model.name == "CBM-classifier-logistic":
            if model.pytorch_mode:
                self.model.to(self.model.device)
                x_train_summary = shap.kmeans(self.model.scores_train, 5)
                self.explainer = shap.KernelExplainer(
                    self.model.infer_proba_from_scores, x_train_summary
                )
            else:
                self.model.backbone.eval()
                x_train_summary = shap.kmeans(self.model.scores_train, 5)
                self.explainer = shap.KernelExplainer(
                    self.model.classifier.predict_proba, x_train_summary
                )

        else:
            ValueError("Model not implemented !")

    def compute_and_plot_explanation(self, image, num_features=6, type_plot="waterfall", **kwargs):
        """Computes SHAP values for a given image and saves the explanation.

        Args:
            image (PIL.Image.Image): The input image.
            num_features (int): The number of features to plot.
            type_plot (str): The type of plot to use. Can be 'waterfall', 'force' or 'bar'.
                Defaults to 'waterfall'.
            kwargs (dict): Additional keyword arguments.
                - save_expl (str): The path to save the explanation image.
        """
        self.model.eval()
        save_expl = kwargs["save_expl"]
        save_activations = kwargs["save_activations"]
        return_activations = kwargs["return_activations"]

        # Get the scores for the input image from the model
        scores = self.model.scores_from_images(image)

        # Compute SHAP explanation for the input image
        d_data_in = {
            self.model.list_concepts[i]: scores[np.newaxis, ...][:, i]
            for i in range(len(self.model.list_concepts))
        }
        pd_data_in = pd.DataFrame(d_data_in)

        shap_values = self.explainer.shap_values(scores, l1_reg=False)

        # Infer the prediction from the model
        prediction = int(self.model.infer_from_scores(scores))
        if save_activations:
            shap_values_abs = np.abs(shap_values[:, prediction])
            attributes_copy = self.model.list_concepts.copy()
            _, shap_values_ordered, l_attributes_ordered = zip(
                *sorted(
                    zip(shap_values_abs, shap_values[:, prediction], attributes_copy, strict=False)
                ),
                strict=False,
            )

            d_data_in_all = {
                l_attributes_ordered[i]: shap_values_ordered[i]
                for i in range(len(shap_values_ordered))
            }
            utils.save_as_json(d_data_in_all, save_activations)

        self.model.train()

        if return_activations:
            shap_values_abs = np.abs(shap_values[:, prediction])
            attributes_copy = self.model.list_concepts.copy()
            _, shap_values_ordered, l_attributes_ordered = zip(
                *sorted(
                    zip(shap_values_abs, shap_values[:, prediction], attributes_copy, strict=False)
                ),
                strict=False,
            )

            return {
                l_attributes_ordered[i]: shap_values_ordered[i]
                for i in range(len(shap_values_ordered))
            }

        # Plot the explanation

        if type_plot == "force":
            # Force variant
            shap.force_plot(
                self.explainer.expected_value[prediction],
                shap_values[:, prediction],
                pd_data_in,
                matplotlib=True,
                show=True,
            )  # for values

        elif type_plot == "bar":
            # Bar variant
            shap.bar_plot(
                shap_values[:, prediction], pd_data_in, show=True, max_display=num_features
            )
            plt.xlabel("Weight value", fontsize=13)

        elif type_plot == "waterfall":
            # Waterfall variant

            shap.plots._waterfall.waterfall_legacy(
                self.explainer.expected_value[prediction],
                shap_values[:, prediction],
                features=scores,
                feature_names=self.model.list_concepts,
                max_display=num_features,
            )

        else:
            ValueError("Plot type not implemented !")

        plt.savefig(save_expl, bbox_inches="tight")

        return None

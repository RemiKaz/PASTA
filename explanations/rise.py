import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from xplique.attributions import Rise as Rise_

import utils

class Rise:
    """Class representing Rise explanations."""

    def __init__(self, model):
        """Initialize the Rise object.

        Parameters:
        ----------
        model: Concept Bottleneck Model to explain.
        """
        super().__init__()

        self.model = model

        # Initialize RISE for Tabular data
        if model.name == "CLIP-QDA":
            self.explainer = Rise_(
                self.model.QDA, grid_size=1, mask_value=self.model.mean_train, nb_samples=4000
            )

        elif model.name == "CBM-classifier-logistic":
            self.model.backbone.eval()
            self.explainer = Rise_(
                self.model.classifier,
                grid_size=1,
                mask_value=self.model.mean_train,
                nb_samples=4000,
            )

        else:
            ValueError("Model not implemented !")

    def compute_and_plot_explanation(self, image, num_features=5, **kwargs):
        """Computes RISE values for a given image and saves the explanation.

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

        # Compute RISE explanation for the input image
        d_data_in = {
            self.model.list_concepts[i]: scores[np.newaxis, ...][:, i]
            for i in range(len(self.model.list_concepts))
        }
        pd_data_in = pd.DataFrame(d_data_in)

        # Infer the prediction from the model
        prediction = int(self.model.infer_from_scores(scores))

        tf_input = tf.cast(scores, tf.float32)
        tf_input = tf.expand_dims(tf_input, axis=0)
        tf_targets = tf.cast(prediction, tf.int32)
        tf_targets = tf.expand_dims(tf_targets, axis=0)

        # one hot encoding of the target
        tf_targets_one = tf.one_hot(tf_targets, len(self.model.list_classes), axis=1)

        vals = self.explainer.explain(tf_input, tf_targets_one)[0].numpy().astype(np.float64)

        # Plot the explanation

        names = self.model.list_concepts.copy()

        d_data_in = {names[i]: vals[i] for i in range(len(vals))}

        # Select only the top num_features entry of the dict

        d_data_in = dict(
            sorted(d_data_in.items(), key=lambda item: abs(item[1]), reverse=True)[:num_features]
        )

        pd_data_in = pd.DataFrame(d_data_in, index=[0])

        self.model.train()

        if save_activations:
            rise_values_abs = np.abs(vals)
            attributes_copy = self.model.list_concepts.copy()
            _, values_ordered, l_attributes_ordered = zip(
                *sorted(zip(rise_values_abs, vals, attributes_copy, strict=False)),
                strict=False,
            )

            d_data_in_all = {
                l_attributes_ordered[i]: values_ordered[i] for i in range(len(values_ordered))
            }

            utils.save_as_json(d_data_in_all, save_activations)

        if return_activations:
            rise_values_abs = np.abs(vals)
            attributes_copy = self.model.list_concepts.copy()
            _, values_ordered, l_attributes_ordered = zip(
                *sorted(zip(rise_values_abs, vals, attributes_copy, strict=False)),
                strict=False,
            )

            return {l_attributes_ordered[i]: values_ordered[i] for i in range(len(values_ordered))}

        sns.set_theme(font_scale=1)

        sns.barplot(
            pd_data_in,
            palette=utils.colors_from_values(list(d_data_in.values()), "coolwarm"),
            orient="h",
        )
        plt.xlabel("Weight value", fontsize=20, fontweight="bold")
        plt.ylabel("Concept", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_expl)

        plt.savefig(save_expl, bbox_inches="tight")

        return None

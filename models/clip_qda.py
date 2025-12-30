import clip
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

import utils


class CLIPQDA(nn.Module):
    def __init__(self, list_concepts, list_classes, device, load_model=False, pytorch_mode=False):
        """Initializes a CLIP Quadratic Discriminant Analysis instance.

        Args:
            list_concepts (list): List of concepts.
            list_classes (list): List of classes.
            device (str): Device to run on.
            load_model (bool, optional): Flag if the model have already been loaded. Defaults to False.
            pytorch_mode (bool, optional): Flag indicating whether to use PyTorch to build the classifier. Defaults to False.
        """
        super().__init__()

        self.name = "CLIP-QDA"
        self.device = device
        self.pytorch_mode = pytorch_mode

        # Initialize Quadratic Discriminant Analysis
        if not load_model:
            self.QDA = QuadraticDiscriminantAnalysis(store_covariance=True)

        if pytorch_mode:
            # dummy mean,cov,priors
            dummy_mean = np.zeros((len(list_classes), len(list_concepts)))
            dummy_var = np.zeros((len(list_classes), len(list_concepts), len(list_concepts)))
            dummy_prior = np.zeros(len(list_classes))
            self.QDA_pytorch = QDAPyTorch(dummy_mean, dummy_var, dummy_prior, self.device)

        # Load CLIP model
        self.clip_net, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_net.eval()

        # Store concepts and classes
        self.list_concepts = list_concepts
        tokens = clip.tokenize(list_concepts).to(self.device)
        self.text_embeding = self.clip_net.encode_text(tokens)
        self.list_classes = list_classes

        # Custom colormap for matplotlib
        colors2 = plt.cm.coolwarm_r(np.linspace(0.5, 1, 128))
        colors1 = plt.cm.coolwarm_r(np.linspace(0, 0.5, 128))
        colors = np.vstack((colors2, colors1))
        mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
        mpl.cm.register_cmap("mycolormap", mymap)

    def train_classifier(self, data_in, data_out, save_train_score=True):
        """Trains the Quadratic Discriminant Analysis model using the given input data and output labels.

        Parameters:
            data_in (numpy.ndarray): Input data for training.
            data_out (numpy.ndarray): Output labels for training.
            save_train_score (bool, optional): Flag indicating whether to save the computed CLIP scores. Defaults to True.

        Returns:
        None
        """
        # Preprocess CLIP scores for training
        clip_scores = self.preprocess_clip_scores(data_in, save_train_score=save_train_score)

        self.mean_train = np.mean(clip_scores).astype(np.float32)  # For occlusion/rise explanations

        # Fit the Quadratic Discriminant Analysis model
        self.QDA.fit(clip_scores, data_out)

        if self.pytorch_mode:
            self.QDA_pytorch = QDAPyTorch(
                self.QDA.means_, self.QDA.covariance_, self.QDA.priors_, self.device
            )

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
        for embeds in tqdm(clip_embeds):
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            text_features = self.text_embeding / self.text_embeding.norm(dim=-1, keepdim=True)
            similarity = (100.0 * embeds @ text_features.T).unsqueeze(0).cpu().detach().numpy()
            clip_scores = np.concatenate((clip_scores, similarity), axis=0)

        # Save score if needed
        if save_train_score:
            self.clip_scores_train = clip_scores

        return clip_scores

    def test(self, data_in, data_out):
        """Computes the accuracy score of the trained Quadratic Discriminant Analysis model on the provided data.

        Parameters:
        data_in (numpy.ndarray): Array of image embeddings.
        data_out (numpy.ndarray): Array of output labels.

        Returns:
        float: The accuracy score of the model.
        """
        # Preprocess CLIP scores for testing
        clip_scores = self.preprocess_clip_scores(data_in)

        # Compute the accuracy score of the trained model

        if self.pytorch_mode:
            return self.QDA_pytorch.score(clip_scores, data_out)

        return self.QDA.score(clip_scores, data_out)

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

        # Predict the label using the trained model
        if self.pytorch_mode:
            similarity = 100.0 * image_features @ text_features.T
            # Forward pass to get log probabilities for each class
            log_probs = self.QDA_pytorch(similarity)
            # Predicted class is the one with the highest log probability
            return torch.argmax(log_probs, dim=1)

        similarity = (100.0 * image_features @ text_features.T).cpu().detach().numpy()

        return self.QDA.predict(similarity)

    def scores_from_images(self, x):
        """Computes the similarity scores between an input image and the text embeddings.

        Parameters:
        input (PIL.Image.Image or np.ndarray): Input image.

        Returns:
        numpy.ndarray: Array of similarity scores.
        """
        # Compute scores for the input image
        img_tensor = self.preprocess(x).unsqueeze(0).to(self.device)
        image_embedding = self.clip_net.encode_image(img_tensor)
        image_features = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        image_features = image_features[0]
        text_features = self.text_embeding / self.text_embeding.norm(dim=-1, keepdim=True)
        return (100.0 * image_features @ text_features.T).cpu().detach().numpy()

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
            return self.QDA_pytorch(input_score_tensor).detach().cpu().numpy()

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

        # Predict the label using the trained model
        if self.pytorch_mode:
            input_score = torch.tensor(input_score).to(self.device)
            # Forward pass to get log probabilities for each class
            log_probs = self.QDA_pytorch(input_score)
            # Predicted class is the one with the highest log probability
            return torch.argmax(log_probs, dim=1).cpu().detach().numpy()

        return self.QDA.predict(input_score)

    def predict_class(self, x, eval_mode=False):
        """Predicts the class of the input images. (Added to homogenize the function predict_class with the other models).

        Parameters:
        x (torch.Tensor): Input images.

        Returns:
        int: Predicted class.
        """
        # Infer labels from input images
        if eval_mode:
            self.clip_net.eval()
            with torch.no_grad():
                output = self.infer_from_images(x)
            self.clip_net.train()
            return output

        return self.infer_from_images(x)

    def forward(self, x):
        """Passes the input through the model and returns the inference result.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Inference result.
        """
        # Pass the input through the model and return the inference result
        return self.infer_from_scores(x)

    def compute_and_plot_conterfactual_explaination_gmm(
        self,
        data,
        top_concepts=5,
        scale_var=True,
        text_explanation=False,
        custom_fig_name=False,
        save_activations=False,
        return_activations=False,
    ):
        """Computes and plots counterfactual explanations for Gaussian Mixture Models.

        Parameters:
        data (numpy.ndarray): Input data.
        top_concepts (int, optional): Number of top concepts to consider. Defaults to 5.
        scale_var (bool, optional): Whether to scale the variance of the GMM. Defaults to True.
        text_explanation (bool, optional): Whether to include text explanation. Defaults to False.
        custom_fig_name (str, optional): Path to save the explanation plot. Defaults to False.
        """
        data = data[np.newaxis, ...]
        i_label_base = self.infer_from_scores(data)[0]  # Infer label from input data

        list_y_not_inf = self.list_classes.copy()  # List of non-inferred labels
        list_y_not_inf.pop(i_label_base)  # Remove the inferred label

        # Dictionaries to store eps, deltas, and lambdas for each concept
        d_eps = {}
        d_deltas = {}
        d_lambdas = {}

        # Compute eps, deltas, and lambdas for each concept
        for i_concept, concept in enumerate(self.list_concepts):
            l_eps = []
            l_deltas = []
            l_lambdas = []
            for labels in list_y_not_inf:
                i_label_j = self.list_classes.index(labels)  # Index of label j
                scores, delta, lambdas = self.calculate_solutions(
                    concept, i_label_base, i_label_j, data
                )
                if scale_var:  # Scale variance if specified
                    if delta < 0:
                        l_eps.append(None)
                    else:
                        if self.pytorch_mode:
                            var = (
                                self.QDA_pytorch.covariances[i_label_base][i_concept, i_concept]
                                .cpu()
                                .detach()
                                .numpy()
                            )
                        else:
                            var = self.QDA.covariance_[i_label_base][i_concept, i_concept]
                        l_eps.append((scores - data[0][i_concept]) / (np.sqrt(var)))
                else:
                    if delta < 0:
                        l_eps.append(None)
                    else:
                        l_eps.append(scores - data[0][i_concept])
                l_deltas.append(delta)
                l_lambdas.append(lambdas)
            d_eps[concept] = l_eps
            d_deltas[concept] = l_deltas
            d_lambdas[concept] = l_lambdas

        # Initialize lists to store counterfactuals and concepts for positive and negative explanations
        l_conterfact_p = []
        l_conterfact_m = []
        l_conterfact_concepts_p = []
        l_conterfact_concepts_m = []
        l_heatmap = np.zeros((2, data.shape[-1]))  # Initialize counterfactual matrix

        # Compute and store counterfactuals and concepts for each concept
        for i, concept in enumerate(self.list_concepts):
            l_eps = d_eps[concept]
            l_deltas = d_deltas[concept]
            l_lambdas = d_lambdas[concept]

            l_eps_ok_i_p_values = []
            l_eps_ok_i_p_label = []
            l_eps_ok_i_m_values = []
            l_eps_ok_i_m_label = []

            for j, eps in enumerate(l_eps):
                if l_deltas[j] > 0:
                    for k, lambda_eps in enumerate(l_lambdas[j]):
                        if lambda_eps > 0 and eps[k] > 0:
                            l_eps_ok_i_p_values.append(eps[k])
                            l_eps_ok_i_p_label.append(list_y_not_inf[j])
                        elif lambda_eps < 0 and eps[k] < 0:
                            l_eps_ok_i_m_values.append(eps[k])
                            l_eps_ok_i_m_label.append(list_y_not_inf[j])

            # Compute and store heatmap, counterfactual, and concept for positive explanations
            if l_eps_ok_i_p_values == []:
                l_heatmap[0, i] = np.inf
                l_conterfact_p.append("No counterfactual")
                l_conterfact_concepts_p.append("No counterfactual")
            else:
                l_heatmap[0, i] = min(l_eps_ok_i_p_values)
                l_conterfact_p.append(
                    f"{concept} (Target = {l_eps_ok_i_p_label[l_eps_ok_i_p_values.index(min(l_eps_ok_i_p_values))]})"
                )
                l_conterfact_concepts_p.append(concept)

            # Compute and store heatmap, counterfactual, and concept for negative explanations
            if l_eps_ok_i_m_values == []:
                l_heatmap[1, i] = -np.inf
                l_conterfact_m.append("No counterfactual")
                l_conterfact_concepts_m.append("No counterfactual")
            else:
                l_heatmap[1, i] = max(l_eps_ok_i_m_values)
                l_conterfact_m.append(
                    f"{concept} (Target = {l_eps_ok_i_m_label[l_eps_ok_i_m_values.index(max(l_eps_ok_i_m_values))]}) "
                )
                l_conterfact_concepts_m.append(concept)

        n_top = top_concepts

        l_concepts_ordered = []

        l_heatmap_ordered = np.empty((1, n_top))  # Initialize ordered heatmap
        heatmap = l_heatmap.reshape(2 * data.shape[-1])  # Reshape heatmap matrix

        # Order the heatmaps and concepts from the lowest to the highest

        l_conterfact_all = l_conterfact_p + l_conterfact_m
        l_conterfact_concepts_all = l_conterfact_concepts_p + l_conterfact_concepts_m

        heatmap_all_abs = np.abs(heatmap)
        _, heatmap, l_conterfact_all, l_conterfact_concepts_all = zip(
            *sorted(
                zip(
                    heatmap_all_abs,
                    heatmap,
                    l_conterfact_all,
                    l_conterfact_concepts_all,
                    strict=False,
                )
            ),
            strict=False,
        )

        if save_activations:
            d_data_in_all = {elem: heatmap[i] for i, elem in enumerate(l_conterfact_all)}
            utils.save_as_json(d_data_in_all, save_activations)

        if return_activations:
            return {elem: heatmap[i] for i, elem in enumerate(l_conterfact_all)}

        l_heatmap_ordered = heatmap[:n_top]
        l_concepts_ordered.append(l_conterfact_all[:n_top])
        labels = l_concepts_ordered

        l_heatmap_ordered = np.array(l_heatmap_ordered)[..., np.newaxis]

        # Remove elements the correspond to an absence of counterfactual
        labels[0] = [elem for elem in labels[0] if elem != "No counterfactual"]
        l_heatmap_ordered = [elem for elem in l_heatmap_ordered if np.abs(elem) != np.inf]

        # Create a DataFrame from the dictionary
        d_data_in = {elem: l_heatmap_ordered[i] for i, elem in enumerate(labels[0])}
        pd_data_in = pd.DataFrame(d_data_in)

        # Plot the exmplanation
        sns.set_theme(font_scale=1)
        sns.barplot(
            pd_data_in,
            palette=utils.colors_from_values(l_heatmap_ordered, "mycolormap"),
            orient="h",
        )

        plt.xlabel("Deviation", fontsize=20, fontweight="bold")
        plt.ylabel("Counterfactual", fontsize=20, fontweight="bold")
        plt.tight_layout()

        # If text_explanation is True, print a message with the explanation
        if text_explanation:
            if l_heatmap_ordered[0] > 0:
                explanation = "By adding some of the concept {} to this image, the model change its label from {} to {}".format(
                    l_conterfact_concepts_all[0],
                    self.list_classes[int(i_label_base)],
                    utils.extract_str(labels[0][0], " =", ")"),
                )
            else:
                explanation = "By removing some of the concept {} to this image, the model change its label from {} to {}".format(
                    l_conterfact_concepts_all[0],
                    self.list_classes[int(i_label_base)],
                    utils.extract_str(labels[0][0], " =", ")"),
                )
            print(explanation)

        plt.savefig(custom_fig_name)
        return None

    def calculate_solutions(self, concept_to_view, label_0, label_j, z_inference):
        """Calculate solutions for counterfactual explanations.

        Parameters:
        - concept_to_view (str): Concept to view.
        - label_0 (int): Label 0.
        - label_j (int): Label j.
        - z_inference (array): Inference data.

        Returns:
        - tuple: Solution, discriminant and lambda values.
        """
        # Get id of the concept to view
        id_to_view = self.list_concepts.index(concept_to_view)

        # Get covariance, means, and priors for labels

        if self.pytorch_mode:
            cov_0 = self.QDA_pytorch.covariances[label_0].cpu().detach().numpy()
            cov_j = self.QDA_pytorch.covariances[label_j].cpu().detach().numpy()
            mean_0 = self.QDA_pytorch.means[label_0].cpu().detach().numpy().transpose()
            mean_j = self.QDA_pytorch.means[label_j].cpu().detach().numpy().transpose()
            p0 = self.QDA_pytorch.priors[label_0].cpu().detach().numpy()
            pj = self.QDA_pytorch.priors[label_j].cpu().detach().numpy()

        else:
            cov_0 = self.QDA.covariance_[label_0]
            cov_j = self.QDA.covariance_[label_j]
            mean_0 = self.QDA.means_[label_0].transpose()
            mean_j = self.QDA.means_[label_j].transpose()
            p0 = self.QDA.priors_[label_0]
            pj = self.QDA.priors_[label_j]

        # Create copy of inference data and set value of concept to view to 0
        z_inference = z_inference.copy()[0].transpose()
        z_inference[id_to_view] = 0

        # Calculate inverse covariances
        inv_cov0 = lg.inv(cov_0)
        inv_covj = lg.inv(cov_j)

        # Calculate P_f_z, b_f_z and c_f_z
        self.P_f_z = 0.5 * (inv_covj - inv_cov0)
        self.b_f_z = (mean_0 - z_inference).transpose() @ inv_cov0 - (
            mean_j - z_inference
        ).transpose() @ inv_covj
        self.c_f_z = 0.5 * (
            np.log(lg.det(cov_j) / lg.det(cov_0))
            + np.log(p0 / pj)
            + (mean_j - z_inference).transpose() @ inv_covj @ (mean_j - z_inference)
            - (mean_0 - z_inference).transpose() @ inv_cov0 @ (mean_0 - z_inference)
        )

        # Calculate discriminant
        disc = self.b_f_z[id_to_view] ** 2 - 4 * self.P_f_z[id_to_view, id_to_view] * self.c_f_z

        # Calculate solutions if discriminant is positive
        if disc < 0:
            return np.array([None, None]), disc, np.array([None, None])

        # Calculate s1, s2 and lambda values
        s1 = (-self.b_f_z[id_to_view] + np.sqrt(disc)) / (2 * self.P_f_z[id_to_view, id_to_view])
        s2 = (-self.b_f_z[id_to_view] - np.sqrt(disc)) / (2 * self.P_f_z[id_to_view, id_to_view])
        lambda1 = (-2 * s1) / (self.b_f_z[id_to_view] + 2 * s1 * self.P_f_z[id_to_view, id_to_view])
        lambda2 = (-2 * s2) / (self.b_f_z[id_to_view] + 2 * s2 * self.P_f_z[id_to_view, id_to_view])

        # Return solutions, discriminant and lambda values
        return np.array([s1, s2]), disc, np.array([lambda1, lambda2])


class QDAPyTorch(nn.Module):
    def __init__(self, means, covariances, priors, device):
        """Initialize the PyTorch version of QDA with the parameters from the sklearn model.

        Args:
            means (np.ndarray): Means of the classes.
            covariances (np.ndarray): Covariance matrices of the classes.
            priors (np.ndarray): Priors for each class.
            device (str): Device to run on.
        """
        super().__init__()

        # Convert means, covariances, and priors to PyTorch tensors
        self.device = device
        self.means = nn.Parameter(torch.tensor(means, dtype=torch.float32))
        self.covariances = nn.Parameter(
            torch.stack([torch.tensor(cov, dtype=torch.float32) for cov in covariances])
        )
        self.priors = nn.Parameter(torch.tensor(priors, dtype=torch.float32))

    def forward(self, x):
        """Compute the log probability of each class for input x.

        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Log probabilities of shape (n_samples, n_classes).
        """
        log_probs = []
        x = x.to(self.device)
        for _i, (mean, cov, prior) in enumerate(zip(self.means, self.covariances, self.priors)):
            # Create multivariate normal distribution for each class
            mvn = MultivariateNormal(mean.to(self.device), covariance_matrix=cov.to(self.device))

            # Compute the log probability for class i
            """#print devices 
            print('mvn.log_prob(x)',mvn.log_prob(x))
            print('torch.log(prior).to(self.device)',torch.log(prior).to(self.device))"""

            log_prob = mvn.log_prob(x) + torch.log(prior).to(self.device)
            log_probs.append(log_prob)

        # Stack the log probabilities for each class and return
        return torch.stack(log_probs, dim=1)

    def score(self, x, y_true):
        """Compute the accuracy of the model by comparing predicted classes with true labels.

        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, n_features).
            y_true (torch.Tensor): True labels tensor of shape (n_samples,).

        Returns:
            float: Mean accuracy of the model.
        """
        # Convert input to tensors
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_true = torch.tensor(y_true, dtype=torch.int64).to(self.device)

        # Forward pass to get log probabilities for each class
        log_probs = self.forward(x)

        # Predicted class is the one with the highest log probability
        y_pred = torch.argmax(log_probs, dim=1)

        # Compute accuracy by comparing predictions with true labels
        correct = (y_pred == y_true).sum().item()
        total = y_true.size(0)

        # Return mean accuracy
        return correct / total

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class PASTA:
    """PASTA-metric model."""

    def __init__(
        self,
        model_name="linear",
        type_task="classification",
        alpha=1,
        device=None,
        input_size=False,
        dict_params=None,
    ):
        """Initializes a classifier to predict scores from criterion.

        Args:
            model_name (str): Name of the classifier model.
            type_task (str): Type of task (classification or regression).
            alpha (float, optional): Alpha value for the lasso and ridge classifier. Defaults to 1.
            device (str, optional): Device to run the model on. Defaults to None.
            input_size (int, optional): Size of the input data. Defaults to False.
            dict_params (dict, optional): Dictionary of parameters to do ablations. Defaults to None.
        """
        self.model_name = model_name
        self.type_task = type_task
        self.device = device

        # Initialize the classifier model based on the provided classifier name
        if model_name == "linear":
            self.classifier = LinearRegression()
            self.name = "PASTA"

        elif model_name == "elastic":
            self.classifier = ElasticNet(alpha=alpha)
            self.name = "PASTA"

        elif model_name == "ridge":
            self.classifier = Ridge(alpha=alpha)
            self.name = "PASTA"

        elif model_name == "lasso":
            self.classifier = Lasso(alpha=alpha)
            self.name = "PASTA"

        elif model_name == "logistic":
            self.classifier = LogisticRegression()
            self.name = "PASTA"

        elif model_name == "QDA":
            self.classifier = QuadraticDiscriminantAnalysis()
            self.name = "PASTA"

        elif model_name == "MLP":
            self.classifier = MLPRegressor(
                hidden_layer_sizes=(),
                activation="identity",
                verbose=True,
                alpha=1e-5,
                batch_size=64,
                learning_rate_init=alpha,
            )
            self.name = "PASTA"
            # Print weights
            """print(self.classifier.coefs_)
            print(self.classifier.coefs_.shape)
            exit()"""

        elif model_name == "svm":
            if type_task == "classification":
                self.classifier = SVC()
            elif type_task == "regression":
                self.classifier = SVR()

            self.name = "PASTA-svm"

        elif model_name == "pytorch":
            self.dict_params = dict_params
            self.classifier = self.PyTorchRegressor(input_size, self.dict_params)
            self.optimizer = optim.Adam(
                self.classifier.parameters(),
                lr=self.dict_params["lr"],
                weight_decay=self.dict_params["w_decay"],
            )
            self.criterion_mse = nn.MSELoss()
            self.criterion_rank = nn.MarginRankingLoss()
            self.name = "PASTA-pytorch"
        else:
            raise ValueError("Classifier not implemented!")

    def calculate_tau_e(self, epoch, total_epochs, gamma):
        """Calculate the value of tau_e based on the current epoch and total epochs.

        Args:
            epoch (int): Current epoch number.
            total_epochs (int): Total number of epochs for training.
            gamma (float): Hyperparameter for controlling the sharpness of the transition.

        Returns:
            float: The value of tau_e for the current epoch.
        """
        return 1 / (1 + np.exp(gamma * (total_epochs / 2 - epoch)))

    '''class PyTorchRegressor(nn.Module):
        """PyTorch model for regression tasks."""

        def __init__(self, input_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 1)

        def forward(self, x):
            return self.fc1(x)'''

    class PyTorchRegressor(nn.Module):
        """PyTorch equivalent for regression tasks."""

        def __init__(self, input_size, dict_params):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(input_size, dict_params["hidden_size"][0]),
                nn.ReLU(),
            )
            if len(dict_params["hidden_size"]) > 1:
                for i in range(1, len(dict_params["hidden_size"])):
                    self.net.append(
                        nn.Linear(dict_params["hidden_size"][i - 1], dict_params["hidden_size"][i])
                    )
                    self.net.append(nn.ReLU())

            self.net.append(nn.Linear(dict_params["hidden_size"][-1], 1))
            self.net.append(nn.ReLU())

        def forward(self, x):
            return self.net(x)

    def train(
        self,
        data_in,
        data_out,
        epochs=400,
        batch_size=16,
        save_score=True,
        data_in_val=False,
        data_out_val=False,
        data_in_test=False,
        data_out_test=False,
    ):
        """Trains the classifier model with the provided input data and output labels.

        Args:
            data_in (numpy.ndarray): Input data for training.
            data_out (numpy.ndarray): Output labels for training.
            epochs (int): Number of epochs for training the PyTorch model.
            batch_size (int): Batch size for PyTorch training.
            save_score (bool): A boolean indicating whether to save the training scores.
            data_in_val (numpy.ndarray, optional): Give input data (instead of using the model) for validation. Defaults to False.
            data_out_val (numpy.ndarray, optional): Give output labels (instead of using the model) for validation. Defaults to False.
            data_in_test (numpy.ndarray, optional): Give input data (instead of using the model) for testing. Defaults to False.
            data_out_test (numpy.ndarray, optional): Give output labels (instead of using the model) for testing. Defaults to False.
        """
        if self.model_name == "pytorch":
            batch_size = self.dict_params["batch_size"]
            epochs = self.dict_params["epochs"]
            alpha_param = self.dict_params["alpha"]
            beta_param = self.dict_params["beta"]
            gamma_param = self.dict_params["gamma"]
            data_in = torch.tensor(data_in, dtype=torch.float32).to(self.device)
            data_out = torch.tensor(data_out, dtype=torch.float32).to(self.device).unsqueeze(1)

            dataset = TensorDataset(data_in, data_out)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            best_qwk_score = -1
            """best_mse_score = 1000"""
            self.classifier.to(self.device)

            for epoch in tqdm(range(epochs)):
                epoch_loss = 0
                epoch_loss_rank = 0
                epoch_loss_sim = 0
                epoch_loss_mse = 0
                """gamma = 0.1
                self.calculate_tau_e(epoch, epochs, gamma)"""
                for inputs, targets in loader:
                    self.optimizer.zero_grad()
                    outputs = self.classifier(inputs)

                    targets = targets.float()

                    # MSE Loss
                    """print(outputs)
                    print(targets)"""

                    loss_mse = self.criterion_mse(outputs, targets)

                    # Ranking loss (targets should be sorted, so we use difference pairs)

                    ranking_loss = self.calculate_ranking_loss(outputs, targets)

                    similarity_loss = 1 - F.cosine_similarity(outputs, targets, dim=1).mean()

                    alpha_ = alpha_param
                    beta_ = beta_param
                    gamma_ = gamma_param
                    """alpha_ = 0.01
                    beta_ = 0.01
                    gamma_ = 0.1"""

                    # Combined loss
                    loss = alpha_ * loss_mse + beta_ * ranking_loss + gamma_ * similarity_loss
                    """loss = tau_e * loss_mse + (1 - tau_e) * ranking_loss"""
                    """loss = 0.5*loss_mse"""
                    loss.backward()

                    """# Test skmearn variant 
                    y_true = outputs.cpu().detach().numpy()
                    y_pred = targets.cpu().detach().numpy()
                    
                    print("sklearn",0.5 * np.average((y_true - y_pred) ** 2, weights=None, axis=0).mean())
                    print("pytorch",loss.item())"""

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_loss_mse += alpha_ * loss_mse.item()
                    epoch_loss_rank += beta_ * ranking_loss.item()
                    epoch_loss_sim += gamma_ * similarity_loss

                if data_in_val is not False and data_out_val is not False and epoch % 5 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss_mse:.4f} {epoch_loss_rank:.4f} {epoch_loss_sim:.4f} {epoch_loss:.4f}"
                    )
                    val_results = self.test(data_in_val, data_out_val)  # Test on validation data
                    val_qwk_score = val_results["Cohen's Kappa"]
                    val_results["MSE"]
                    # Save the best model based on QWK score
                    if val_qwk_score > best_qwk_score:
                        best_qwk_score = val_qwk_score
                        best_model_state = copy.deepcopy(
                            self.classifier.state_dict()
                        )  # Save model state
                        # Save as pth
                        print(f"New Best QWK Score: {best_qwk_score:.4f} at epoch {epoch+1}")
                        test_results = self.test(
                            data_in_test, data_out_test
                        )  # Test on validation data
                        test_cohen_kappa = test_results["Cohen's Kappa"]
                        print(
                            f"Test MSE: {test_results['MSE']:.4f}, Test QWK: {test_cohen_kappa:.4f}"
                        )
                    else:
                        print(
                            f"Validation QWK Score: {val_qwk_score:.4f} (Best: {best_qwk_score:.4f})"
                        )
                    """if val_mse_score < best_mse_score:
                        best_mse_score = val_mse_score
                        best_model_state = self.classifier.state_dict()  # Save model state
                        # Save as pth
                        print(f"New Best MSE Score: {best_mse_score:.4f} at epoch {epoch+1}")
                    else:
                        print(
                            f"Validation MSE Score: {val_mse_score:.4f} (Best: {best_mse_score:.4f})"
                        )"""

            # At the end of training, save or load the best model
            if best_model_state is not None:
                # Print actual parameters
                self.classifier.load_state_dict(best_model_state)  # Load the best model
                self.best_model_state = best_model_state
                # Print parameters
                """print(self.classifier.fc1.weight)   
                print(self.classifier.fc1.bias)
                print(self.classifier.fc1.weight.size())
                print(self.classifier.fc1.bias.size()) """
        else:
            self.classifier.fit(data_in, data_out)
            # Print params
            """print(self.classifier.coefs_)
            print(self.classifier.intercepts_)
            print(np.array(self.classifier.coefs_).shape)
            print(np.array(self.classifier.intercepts_).shape)"""

        if save_score:
            self.scores_train = data_in

    def calculate_ranking_loss(self, outputs, targets):
        """Computes the ranking loss based on the hinge loss between pairs."""
        """differences = (outputs[:-1] - outputs[1:])
        """

        target_differences = targets[:-1] - targets[1:]
        return self.criterion_rank(outputs[:-1], outputs[1:], torch.sign(target_differences))

    def test(self, data_in, data_out, given_label=False):
        """Compute the accuracy score of the trained classifier.

        Args:
            data_in (numpy.ndarray): Input data for testing.
            data_out (numpy.ndarray): Output labels for testing.
            given_label (bool, optional): Whether to use given label (instead of making a prediction) or not.

        Returns:
            dict: Dictionary containing various evaluation metrics.
        """
        if given_label:
            # Calculate mean sqare error
            mse = mean_squared_error(data_out, data_in)

            # Calculate R2 score
            ss_res = np.sum((data_in - data_out) ** 2)
            ss_tot = np.sum((data_out - np.mean(data_out)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            # Calculate pseudo-classification accuracy
            pseudo_accuracy = self.pseudo_classification_score(data_in, data_out, given_label=True)

            # Calculate Cohen's Kappa score
            qwk_score = self.quadratic_weighted_kappa_score(data_in, data_out, given_label=True)

            # Calculate spearman_coef
            spearman_coef_score = self.spearman_coef(data_in, data_out, given_label=True)

            return {
                "MSE": mse,
                "R2": r2,
                "Pseudo-Classification Accuracy": pseudo_accuracy,
                "Cohen's Kappa": qwk_score,
                "Spearman Coef": spearman_coef_score,
            }

        # If using a PyTorch model
        if self.model_name == "pytorch":
            self.classifier.eval()  # Set model to evaluation mode

            data_in = torch.tensor(data_in, dtype=torch.float32).to(self.device)
            data_out = torch.tensor(data_out, dtype=torch.float32).to(self.device).unsqueeze(1)

            with torch.no_grad():
                predictions = self.classifier(data_in).cpu().numpy()

            # Clip the predictions to the range [0, 1]
            predictions = np.clip(predictions, 0, 1)

            """print(predictions)"""

            # Calculate MSE
            mse = mean_squared_error(data_out.cpu(), predictions)

            # Calculate R2 score
            ss_res = np.sum((predictions - data_out.cpu().numpy()) ** 2)
            ss_tot = np.sum((data_out.cpu().numpy() - np.mean(data_out.cpu().numpy())) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            # Calculate pseudo-classification accuracy
            pseudo_accuracy = self.pseudo_classification_score(
                data_in.cpu().numpy(), data_out.cpu().numpy()
            )

            # Calculate Cohen's Kappa score
            qwk_score = self.quadratic_weighted_kappa_score(
                data_in.cpu().numpy(), data_out.cpu().numpy()
            )

            # Calculate spearman_coef
            spearman_coef_score = self.spearman_coef(data_in, data_out)

            self.classifier.train()

            return {
                "MSE": mse,
                "R2": r2,
                "Pseudo-Classification Accuracy": pseudo_accuracy,
                "Cohen's Kappa": qwk_score,
                "Spearman Coef": spearman_coef_score,
            }

        # Fallback for non-pytorch models (same as before)
        predictions = []
        for i in range(len(data_in)):
            prediction = self.classifier.predict([data_in[i]])
            predictions.append(np.clip(prediction, 0, 1))

        # Calculate MSE
        mse = mean_squared_error(data_out, predictions)

        # Calculate R2 score
        r2 = self.classifier.score(data_in, data_out)

        # Calculate pseudo-classification accuracy
        pseudo_accuracy = self.pseudo_classification_score(data_in, data_out)

        # Calculate Cohen's Kappa score
        qwk_score = self.quadratic_weighted_kappa_score(data_in, data_out)

        # Calculate spearman_coef
        spearman_coef_score = self.spearman_coef(data_in, data_out)

        return {
            "MSE": mse,
            "R2": r2,
            "Pseudo-Classification Accuracy": pseudo_accuracy,
            "Cohen's Kappa": qwk_score,
            "Spearman Coef": spearman_coef_score,
        }

    def pseudo_classification_score(
        self, data_in, data_out, interval_radius=0.5, given_label=False
    ):
        """Compute pseudo-classification accuracy based on interval matching.

        Args:
            data_in (numpy.ndarray): Input data for testing.
            data_out (numpy.ndarray): True output labels for testing.
            interval_radius (float): Radius around target integer for considering a correct prediction.
            given_label (bool, optional): Whether to use given label (instead of making a prediction) or not.

        Returns:
            float: The pseudo-classification accuracy score.
        """
        correct_predictions = 0
        total_predictions = len(data_in)

        # Define target intervals around integers 1, 2, 3, 4, 5
        target_intervals = {
            1: (1 - interval_radius, 1 + interval_radius),
            2: (2 - interval_radius, 2 + interval_radius),
            3: (3 - interval_radius, 3 + interval_radius),
            4: (4 - interval_radius, 4 + interval_radius),
            5: (5 - interval_radius, 5 + interval_radius),
        }

        for i in range(len(data_in)):
            if given_label:
                prediction = data_in[i] * 4 + 1
            elif self.model_name == "pytorch":
                data_in_tensor = (
                    torch.tensor(data_in[i], dtype=torch.float32).to(self.device).unsqueeze(0)
                )
                prediction = (
                    np.clip(self.classifier(data_in_tensor).cpu().detach().numpy()[0], 0, 1) * 4 + 1
                )
            else:
                prediction = np.clip(self.classifier.predict([data_in[i]]), 0, 1) * 4 + 1

            true_value = data_out[i] * 4 + 1

            """if i < 10:
                print(f"Prediction: {prediction}, True Value: {true_value}")"""

            # Find the interval that matches the true value
            matched_interval = None
            for interval in target_intervals.values():
                if interval[0] <= true_value <= interval[1]:
                    matched_interval = interval
                    break

            for target_value, interval in target_intervals.items():
                if true_value == target_value:
                    matched_interval = interval
                    break

            if (matched_interval is not None) and (
                matched_interval[0] <= prediction <= matched_interval[1]
            ):
                correct_predictions += 1

            """print(f"Prediction: {prediction}, True Value: {true_value}, Interval: {matched_interval}")"""

        return correct_predictions / total_predictions

    def quadratic_weighted_kappa_score(
        self, data_in, data_out, interval_radius=0.5, given_label=False
    ):
        """Compute quadratic weighted kappa score based on interval matching.

        Args:
            data_in (numpy.ndarray): Input data for testing.
            data_out (numpy.ndarray): True output labels for testing.
            interval_radius (float): Radius around target integer for considering a correct prediction.
            given_label (bool, optional): Whether to use given label (instead of making a prediction) or not.

        Returns:
            float: The pseudo-classification accuracy score.
        """
        # Define target intervals around integers 1, 2, 3, 4, 5
        target_intervals = {
            1: (1 - interval_radius, 1 + interval_radius),
            2: (2 - interval_radius, 2 + interval_radius),
            3: (3 - interval_radius, 3 + interval_radius),
            4: (4 - interval_radius, 4 + interval_radius),
            5: (5 - interval_radius, 5 + interval_radius),
        }

        l_pseudo_predict = []
        l_label = []

        for i in range(len(data_in)):
            if given_label:
                prediction = data_in[i] * 4 + 1
            elif self.model_name == "pytorch":
                data_in_tensor = (
                    torch.tensor(data_in[i], dtype=torch.float32).to(self.device).unsqueeze(0)
                )
                prediction = (
                    np.clip(self.classifier(data_in_tensor).cpu().detach().numpy()[0], 0, 1) * 4 + 1
                )
            else:
                prediction = np.clip(self.classifier.predict([data_in[i]]), 0, 1) * 4 + 1
            true_value = data_out[i] * 4 + 1
            # Find the interval that matches the true value
            pred_pseudo_label = None
            true_pseudo_label = None
            for value, interval in target_intervals.items():
                if interval[0] <= prediction <= interval[1]:
                    pred_pseudo_label = value
                if interval[0] <= true_value <= interval[1]:
                    true_pseudo_label = value

            if pred_pseudo_label is None or true_pseudo_label is None:
                ValueError("No matching interval found")

            l_pseudo_predict.append(pred_pseudo_label)
            l_label.append(true_pseudo_label)

        return cohen_kappa_score(l_pseudo_predict, l_label, weights="quadratic")

    def spearman_coef(self, data_in, data_out, interval_radius=0.5, given_label=False):
        """Compute quadratic weighted kappa score based on interval matching.

        Args:
            data_in (numpy.ndarray): Input data for testing.
            data_out (numpy.ndarray): True output labels for testing.
            interval_radius (float): Radius around target integer for considering a correct prediction.
            given_label (bool, optional): Whether to use given label (instead of making a prediction) or not.

        Returns:
            float: The pseudo-classification accuracy score.
        """
        # Define target intervals around integers 1, 2, 3, 4, 5
        target_intervals = {
            1: (1 - interval_radius, 1 + interval_radius),
            2: (2 - interval_radius, 2 + interval_radius),
            3: (3 - interval_radius, 3 + interval_radius),
            4: (4 - interval_radius, 4 + interval_radius),
            5: (5 - interval_radius, 5 + interval_radius),
        }

        l_pseudo_predict = []
        l_label = []

        for i in range(len(data_in)):
            if given_label:
                prediction = data_in[i] * 4 + 1
            elif self.model_name == "pytorch":
                data_in_tensor = (
                    torch.tensor(data_in[i], dtype=torch.float32).to(self.device).unsqueeze(0)
                )
                prediction = (
                    np.clip(self.classifier(data_in_tensor).cpu().detach().numpy()[0], 0, 1) * 4 + 1
                )
            else:
                prediction = np.clip(self.classifier.predict([data_in[i]]), 0, 1) * 4 + 1
            true_value = data_out[i] * 4 + 1

            # Find the interval that matches the true value
            pred_pseudo_label = None
            true_pseudo_label = None
            for value, interval in target_intervals.items():
                if interval[0] <= prediction <= interval[1]:
                    pred_pseudo_label = value
                if interval[0] <= true_value <= interval[1]:
                    true_pseudo_label = value

            if pred_pseudo_label is None or true_pseudo_label is None:
                ValueError("No matching interval found")

            l_pseudo_predict.append(pred_pseudo_label)
            l_label.append(true_pseudo_label)

        return spearmanr(l_pseudo_predict, l_label)

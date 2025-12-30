import clip
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class CLIPClassifierProbe:
    def __init__(self, list_classes, device, classifier="logistic"):
        """Initializes a CLIP classifier instance.

        Args:
            list_classes (list): List of classes.
            device (str): Device to run on.
            classifier (str, optional): Name of the classifier. Defaults to "logistic".
        """
        self.name = "CLIP-classifier-probe"
        self.device = device

        # Initialize classifier
        if classifier == "logistic":
            self.classifier = LogisticRegression()

        elif classifier == "qda":
            self.classifier = QuadraticDiscriminantAnalysis()

        elif classifier == "svm":
            self.classifier = SVC()

        # Load CLIP model
        self.clip_net, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_net.eval()

        # Store concepts and classes
        self.list_classes = list_classes

        # Custom colormap for matplotlib
        colors2 = plt.cm.coolwarm_r(np.linspace(0.5, 1, 128))
        colors1 = plt.cm.coolwarm_r(np.linspace(0, 0.5, 128))
        colors = np.vstack((colors2, colors1))
        mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
        mpl.cm.register_cmap("mycolormap", mymap)

    def train(self, data_in, data_out, save_train_embedings=True):
        """Trains the classifier model using the given input data and output labels.

        Parameters:
            data_in (numpy.ndarray): Input data for training.
            data_out (numpy.ndarray): Output labels for training.
            save_train_embedings (bool, optional): Flag indicating whether to save the computed CLIP embedings. Defaults to True.

        Returns:
        None
        """
        # Preprocess CLIP embedings for training
        image_embedings = self.preprocess_clip_embedings(
            data_in, save_train_embedings=save_train_embedings
        )

        # Fit the classifier model
        self.classifier.fit(image_embedings.astype(np.float32), data_out)

    def preprocess_clip_embedings(self, b_images, verbose=True, save_train_embedings=False):
        """Computes the CLIP image embedings for the given images.

        Parameters:
        images: Array of images.
        verbose (bool, optional): Flag indicating whether to print a progress message. Defaults to True.
        save_train_embedings (bool, optional): Flag indicating whether to save the computed CLIP embedings. Defaults to False.

        Returns:
        numpy.ndarray: Array of CLIP embedings.
        """
        if verbose:
            print("Computing CLIP embedings...")

        b_images_rgb = np.repeat(b_images[:, :, :, np.newaxis], 3, axis=3)

        # Convert the numpy array to a PIL Image and then apply CLIP preprocessing
        images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in b_images_rgb]

        # Apply CLIP preprocessing to each image
        images_preprocessed = torch.stack(
            [self.preprocess(img).to(self.device) for img in images_pil]
        )

        # Compute the CLIP embeddings
        with torch.no_grad():
            return self.clip_net.encode_image(images_preprocessed).cpu().detach().numpy()

    def test(self, data_in, data_out):
        """Computes the accuracy score of the trained classifier model on the provided data.

        Parameters:
        data_in (numpy.ndarray): Array of image embeddings.
        data_out (numpy.ndarray): Array of output labels.

        Returns:
        float: The accuracy score of the model.
        """
        # Preprocess CLIP embedings for testing
        clip_embedings = self.preprocess_clip_embedings(data_in)

        """print(self.classifier.predict(clip_embedings),data_out)"""

        # Compute the accuracy score of the trained model
        return self.classifier.score(clip_embedings, data_out)

    def infer_from_images(self, x):
        """Given an input image, predicts the label using the trained model.

        Parameters:
        input (PIL.Image.Image or np.ndarray): Input image.

        Returns:
        int: Predicted label.
        """
        # Compute embedings for the input image
        img_tensor = self.preprocess(x).unsqueeze(0).to(self.device)
        image_embedding = self.clip_net.encode_image(img_tensor).cpu().detach().numpy()

        # Predict the label using the trained model
        return self.classifier.predict(image_embedding)

    def embedings_from_images(self, x):
        """Computes the similarity embedings between an input image and the text embeddings.

        Parameters:
        input (PIL.Image.Image or np.ndarray): Input image.

        Returns:
        numpy.ndarray: Array of similarity embedings.
        """
        # Compute embedings for the input image
        img_tensor = self.preprocess(x).unsqueeze(0).to(self.device)
        image_embedding = self.clip_net.encode_image(img_tensor)

        return image_embedding.cpu().detach().numpy()

    def infer_from_embedings(self, input_embedings):
        """Given input similarity embedings, predicts the label using the trained model.

        Parameters:
        input_embedings (numpy.ndarray): Array of similarity embedings.

        Returns:
        int: Predicted label.
        """
        input_embedings = input_embedings.reshape(1, -1)

        # Predict the label using the trained model
        return self.classifier.predict(input_embedings)

    def infer_proba_from_embedings(self, input_embedings):
        """Given input similarity embedings, predicts the label probabilities using the trained model.

        Parameters:
        input_embedings (numpy.ndarray): Array of similarity embedings.

        Returns:
        numpy.ndarray: Array of label probabilities.
        """
        input_embedings = input_embedings.reshape(1, -1)

        # Predict the label probabilities using the trained model
        return self.classifier.predict_proba(input_embedings)

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
        return self.infer_from_embedings(x)

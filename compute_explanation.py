import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import data
import explanations
import models
import utils


def make_plots_explanation(
    model,
    input_image,
    explanation_method,
    name_img=None,
    save_img=None,
    save_expl=None,
    save_activations=None,
    return_activations=False,
):
    """Function to make plots for explanation and save them if specified.

    Args:
        model (nn.Module or torch.nn.Module): Model to predict class.
        input_image (numpy.ndarray): Input image.
        explanation_method (ExplanationMethod): Explanation method to use.
        save_img (str, optional): Path to save input image. Defaults to None.
        save_expl (str, optional): Path to save explanation. Defaults to None.
        save_activations (str, optional): Path to save activations. Defaults to None.
        return_activations (bool, optional): Whether to return activations. Defaults to False.
        name_img (str, optional): Name of the image. Defaults to None.

    Returns:
        int: Predicted class id.
    """
    """# Set model to evaluation mode if it's a nn.Module
    if isinstance(model, nn.Module):
        model.eval()"""

    # Save input image if specified
    if save_img:
        input_image_pil = Image.fromarray(np.uint8(input_image))
        input_image_pil.save(save_img)

    if name_img:
        # Compute and plot explanation
        explanation_method.compute_and_plot_explanation(
            input_image,
            model=model,
            save_expl=save_expl,
            save_activations=save_activations,
            return_activations=return_activations,
            name_img=name_img,
        )
    else:
        # Compute and plot explanation
        explanation_method.compute_and_plot_explanation(
            input_image,
            model=model,
            save_expl=save_expl,
            save_activations=save_activations,
            return_activations=return_activations,
        )
    plt.close()

    # Predict class id
    return model.predict_class(input_image, eval_mode=True)[0]


if __name__ == "__main__":
    ## Parser
    def parse_args():
        # Training settings
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "--network",
            type=str,
            default="resnet50",
            help="network (resnet50, vitB, CLIP-QDA, CLIP-zero-shot, CLIP-Linear, CBM-classifier-logistic, CLIP-LaBo, XNES-classifier-logistic)",
        )
        parser.add_argument("--gpus", type=str, default="0", help="gpu ids")
        parser.add_argument(
            "--expl_method",
            type=str,
            default="GradCAM",
            help="explanation method (Rise_PASTA ,Occlusion_CBM, Rise_CBM, LIME, SHAP, GradCAM, CLIP-QDA-sample, LIME_CBM, SHAP_CBM, AblationCAM, EigenCAM, EigenGradCAM, FullGrad, GradCAMPlusPlus, GradCAMElementWise, HiResCAM, ScoreCAM, XGradCAM, DeepFeatureFactorization, CLIP-Linear-sample, CLIP-LaBo-sample, Xnesyl-Linear)",
        )
        parser.add_argument(
            "--input_dataset",
            type=str,
            default="monumai",
            help="set of input data to use in [pascalpart,monumai,coco,catsdogscars]",
        )
        parser.add_argument(
            "--input_image_path",
            type=str,
            default="data_samples/Monumai/Images/MonuMAI_6_Baroque.png",
            help="for sample wise explanations, input image",
        )

        parser.add_argument(
            "--id_perturbation",
            type=int,
            default=-1,
            help="Id of the perturbation, if -1 no perturbation is applied",
        )

        parser.add_argument(
            "--lambda_pasta",
            type=float,
            default=0.5,
            help="Id of the perturbation, if -1 no perturbation is applied",
        )

        return parser.parse_args()

    args = parse_args()

    # Load data/explainer/model
    args = parse_args()
    device = f"cuda:{args.gpus}" if (torch.cuda.is_available() and args.gpus != "-1") else "cpu"

    metadata = data.metadata_importer(args.input_dataset)

    if args.expl_method == "BCos":
        model = models.model_importer(
            args.network, metadata, device, load_model=True, bcos_eval=True
        )

    else:
        model = models.model_importer(args.network, metadata, device, load_model=True,pytorch_mode=True)

    explanation_method = explanations.explanation_method_importer(
        args.expl_method,
        device=device,
        model=model,
        model_name=args.network,
        metadata=metadata,
        lambda_pasta=args.lambda_pasta,
    )

    if args.input_image_path.endswith(".png"):  # Read image paths from a text file
        image_paths = [args.input_image_path]
    else:  # List of all images in folder
        image_paths = list(Path(args.input_image_path).glob("*.png"))

    for imgs_pths in image_paths:
        print("image selected", imgs_pths)

        input_image = Image.open(imgs_pths).convert("RGB")

        # Perturbate image
        if args.id_perturbation != -1:
            image_perturbator = utils.ImagePerturbator()
            input_image_pert = image_perturbator.perturbate_image(
                input_image, args.id_perturbation, 7
            )
            input_image = input_image_pert

        # If explanation as concepts importance, save as json if concepts as saliency map, save as npy
        if args.expl_method in [
            "CLIP-QDA-sample",
            "LIME_CBM",
            "SHAP_CBM",
            "Xnesyl-Linear",
            "CLIP-Linear-sample",
            "CLIP-LaBo-sample",
            "Occlusion-cbm",
        ]:
            save_activations_path = (
                "results/out_activations/" + str(imgs_pths).replace(".png", ".json").split("/")[-1]
            )
            save_explanation_path = (
                "results/out_explanations/" + str(imgs_pths).replace(".png", ".png").split("/")[-1]
            )

        else:
            save_activations_path = (
                "results/out_activations/" + str(imgs_pths).replace(".png", ".npy").split("/")[-1]
            )
            save_explanation_path = (
                "results/out_explanations/" + str(imgs_pths).replace(".png", ".png").split("/")[-1]
            )

        # Compute explanations from PIL images
        if args.expl_method == "Rise_PASTA":
            label = make_plots_explanation(
                model,
                input_image,
                explanation_method,
                name_img=imgs_pths,
                save_img="results/img.png",
                save_expl=f"results/optims_pasta/{imgs_pths.split('/')[-1].split('.')[0]}_{str(args.lambda_pasta).replace('.','')}.png",
                save_activations=f"results/optims_pasta_activations/{imgs_pths.split('/')[-1].split('.')[0]}_{str(args.lambda_pasta).replace('.','')}.npy",
            )
        else:
            label = make_plots_explanation(
                model,
                input_image,
                explanation_method,
                save_img="results/img.png",
                save_expl=save_explanation_path,
                save_activations=save_activations_path,
            )

        print("label", metadata["labels"][label])

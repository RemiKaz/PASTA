import argparse

import numpy as np
import open_clip
import torch
from PIL import Image

import models
import utils
from compute_explanation import make_plots_explanation
from pathlib import Path
import json

def parse_args():
    # Argument settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids")
    parser.add_argument(
        "--path_explanation",
        type=str,
        help="Path to the explanation image directory",
    )
    parser.add_argument(
        "--id_question",
        type=str,
        default="Q1",
        help="For the id of the question in [Q1, Q2, Q3, Q4, Q5, Q6]",
    )
    parser.add_argument(
        "--out_format",
        type=str,
        default="saliency",
        help="indicate if the explanation is either a saliency map or a concept based explanation",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Device setup
    device = f"cuda:{args.gpus}" if (torch.cuda.is_available() and args.gpus != "-1") else "cpu"

    dict_params = {
        "batch_size": 256,
        "epochs": 600,
        "alpha": 1,
        "beta": 0.001,
        "gamma": 0.01,
        "lr": 0.000002,
        "w_decay": 1e-6,
        "optimizer": "Adam",
        "template": "",
        "N_top": 20,
        "hidden_size": [512, 64],
        "file_annotations": "human_annotations.json",
    }

    if args.dataset == None :
        print("The explanation must come from the test set of coco,pascalpart,monumai,catsdogscars (sorry)")

    # Load PASTA model
    model_pasta = models.model_importer(
        "PASTA-pytorch",
        None,
        device,
        type_task="regression",
        alpha=1,
        input_size=1050,
        dict_params=dict_params,
    )
    state_dict = torch.load(f"models_pkl/PASTA_{args.id_question}.pth")
    model_pasta.classifier.load_state_dict(state_dict)
    model_pasta.classifier.to(device)

    # Load CLIP model
    model_name = "ViT-L-16-SigLIP-384"
    pretrained = "webli"
    openclip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
    openclip_model.eval().to(device)

    # List PASTA_scores
    pasta_scores = []

    if args.out_format == "saliency":
            
        for path_explanation in Path(args.path_explanation).glob("*"):

            # Label name
            label = path_explanation.split("_")[-1].split(".")[0]
            one_hot_label = utils.convert_label_to_one_hot(label)

            explication = np.array(Image.open(path_explanation))
            img_tensor = preprocess(Image.fromarray(np.uint8(explication))).unsqueeze(0).to(device)
            embedding = openclip_model.encode_image(img_tensor).float()
            in_pasta = np.concatenate((one_hot_label, embedding.cpu().detach().numpy().squeeze(0)))
            in_pasta = torch.tensor(in_pasta).to(device).float()
            score = model_pasta.classifier(in_pasta).cpu().detach().numpy()[0] * 4 + 1
            pasta_scores.append(score)

    elif args.out_format == "cbm":

        tokenizer = open_clip.get_tokenizer(model_name)

        for path_explanation in Path(args.path_explanation).glob("*"):

            # Label name
            label = path_explanation.split("_")[-1].split(".")[0]
            one_hot_label = utils.convert_label_to_one_hot(label)

            # Dataset name
            dataset_name = path_explanation.split("/")[-1].split("_")[0]

            with Path(path_explanation).open("r") as fp:
                activations_dict = json.load(fp)

            activations = utils.activations_from_dict(
                activations_dict, dataset_name, "0"
            )
            text = utils.sentence_from_activations(
                activations, dataset_name, "0", top_n=15
            )
            tokens = tokenizer(text).to(device)
            text_embedding = (
                openclip_model.encode_text(tokens).cpu().detach().numpy().squeeze(0)
            )
            # Concatenate one_hot_label and text_embedding
            in_pasta = np.concatenate((one_hot_label, text_embedding.cpu().detach().numpy().squeeze(0)))
            in_pasta = torch.tensor(in_pasta).to(device).float()
            score = model_pasta.classifier(in_pasta).cpu().detach().numpy()[0] * 4 + 1
            pasta_scores.append(score)

    else:
        raise ValueError("Unknown output format")
    
    print(f"PASTA score of the method: {np.mean(pasta_scores):.2f}")


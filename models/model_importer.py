import torch

import data
from models.cbm_supervised import CBMsupervised
from models.clip_few_shot_probe import CLIPClassifierProbe
from models.clip_labo import CLIPLaBo
from models.clip_linear import CLIPLinear
from models.clip_qda import CLIPQDA
from models.clip_zero_shot import CLIPzeroshot
from models.pasta import PASTA
from models.resnet import Resnet50
from models.vit import VitB
from models.xnesyl import Xnesyl
from train_net import train_sklearn


def model_importer(
    model_name,
    metadata_dataset,
    device,
    load_model=False,
    type_task="classification",
    bcos_eval=False,
    alpha=1,
    input_size=False,
    pytorch_mode=False,
    dict_params=None,
    model_clip="ViT-B/32",  # Added for compatibility with clip-zero-shot
    pretrained=None,  # Added for compatibility with clip-zero-shot
):
    """Imports the specified model and optionally loads its weights.

    Args:
        model_name (str): The name of the model to import.
        metadata_dataset (dict): The metadata of the dataset.
        device (str): The device to run the model on.
        load_model (bool, optional): Whether to load the model weights. Defaults to False.
        type_task (str, optional): The type of task. Defaults to "classification".
        bcos_eval (bool, optional): Whether to use bcos for evaluation. Defaults to False.
        alpha (float, optional): Alpha value for the lasso and ridge classifier. Defaults to 1.
        input_size (int, optional): Size of the input data. Defaults to False.
        pytorch_mode (bool, optional): Whether to use pytorch to build the model. Defaults to False.
        dict_params (dict, optional): Dictionary of parameters to do ablations. Defaults to None.
        model_clip (str, optional): The CLIP model to use. Defaults to "ViT-B/32".
        pretrained (str, optional): Pretrained weights for CLIP-zero-shot. Defaults to None.

    Returns:
        torch.nn.Module or object: The imported model.

    Raises:
        Exception: If the model is not implemented.
    """
    print(f"Importing model {model_name} ...")

    # Import model, for CBMs, we use labeled, unoptimal concepts
    if model_name == "resnet50":
        model = Resnet50(len(metadata_dataset["labels"]), device)
    elif model_name == "resnet50-bcos":
        model = Resnet50(len(metadata_dataset["labels"]), device, bcos=True, bcos_eval=bcos_eval)
    elif model_name == "CLIP-few-shot-qda":
        model = CLIPClassifierProbe(metadata_dataset["labels"], device, classifier="qda")
    elif model_name == "CLIP-few-shot-logistic":
        model = CLIPClassifierProbe(metadata_dataset["labels"], device, classifier="logistic")
    elif model_name == "CLIP-few-shot-svm":
        model = CLIPClassifierProbe(metadata_dataset["labels"], device, classifier="svm")
    elif model_name == "vitB":
        model = VitB(len(metadata_dataset["labels"]), device)
    elif model_name == "vitB-bcos":
        model = VitB(len(metadata_dataset["labels"]), device, bcos=True, bcos_eval=bcos_eval)
    elif model_name == "CLIP-QDA":
        model = CLIPQDA(
            metadata_dataset["labeled_concepts"],
            metadata_dataset["labels"],
            device,
            pytorch_mode=pytorch_mode,
        )
    elif model_name == "CLIP-zero-shot":
        model = CLIPzeroshot(metadata_dataset["labels"], device, model_clip, pretrained)
    elif model_name == "CLIP-LaBo":
        model = CLIPLaBo(metadata_dataset["labeled_concepts"], metadata_dataset["labels"], device)
    elif model_name == "CLIP-Linear":
        model = CLIPLinear(
            metadata_dataset["labeled_concepts"],
            metadata_dataset["labels"],
            device,
        )
    elif model_name == "CBM-backbone-resnet":
        cbm = CBMsupervised(
            metadata_dataset["labeled_concepts"],
            metadata_dataset["labels"],
            device,
            backbone_name="resnet",
            classifier_name="linear",
            pytorch_mode=pytorch_mode,
        )
        model = cbm.backbone
    elif model_name == "CBM-classifier-linear":
        model = CBMsupervised(
            metadata_dataset["labeled_concepts"],
            metadata_dataset["labels"],
            device,
            backbone_name="resnet",
            classifier_name="linear",
            pytorch_mode=pytorch_mode,
        )
    elif model_name == "CBM-classifier-logistic":
        model = CBMsupervised(
            metadata_dataset["labeled_concepts"],
            metadata_dataset["labels"],
            device,
            backbone_name="resnet",
            classifier_name="logistic",
            pytorch_mode=pytorch_mode,
        )
    elif model_name == "XNES-backbone-faster-rcnn":
        model = Xnesyl(
            metadata_dataset["labeled_concepts"],
            metadata_dataset["labels"],
            device,
            pytorch_mode=pytorch_mode,
        )
        model = model.backbone
    elif model_name == "XNES-classifier-logistic":
        model = Xnesyl(
            metadata_dataset["labeled_concepts"],
            metadata_dataset["labels"],
            device,
            classifier_name="logistic",
            pytorch_mode=pytorch_mode,
        )
    elif model_name == "PASTA-logistic":
        model = PASTA(model_name="logistic", type_task=type_task)
    elif model_name == "PASTA-linear":
        model = PASTA(model_name="linear", type_task=type_task)
    elif model_name == "PASTA-QDA":
        model = PASTA(model_name="QDA", type_task=type_task)
    elif model_name == "PASTA-svm":
        model = PASTA(model_name="svm", type_task=type_task)
    elif model_name == "PASTA-elastic":
        model = PASTA(model_name="elastic", type_task=type_task, alpha=alpha)
    elif model_name == "PASTA-ridge":
        model = PASTA(model_name="ridge", type_task=type_task, alpha=alpha)
    elif model_name == "PASTA-lasso":
        model = PASTA(model_name="lasso", type_task=type_task, alpha=alpha)
    elif model_name == "PASTA-mlp":
        model = PASTA(model_name="MLP", type_task=type_task, alpha=alpha)
    elif model_name == "PASTA-pytorch":
        model = PASTA(
            model_name="pytorch",
            type_task=type_task,
            alpha=alpha,
            input_size=input_size,
            dict_params=dict_params,
        )
    else:
        raise ValueError("Network not implemented !")

    # Load model
    if load_model:
        if (
            model_name == "resnet50"
            or model_name == "vitB"
            or model_name == "CLIP-LaBo"
            or model_name == "CLIP-Linear"
            or model_name == "resnet50-bcos"
            or model_name == "vitB-bcos"
        ):
            # Bad code to deal with change of the class name

            ckpt = torch.load(
                "/models_pkl/best_{}_{}.pth".format(model_name, metadata_dataset["name"]),weights_only=False
            )

            if isinstance(ckpt, dict):
                model.load_state_dict(ckpt)

            else:
                model.load_state_dict(ckpt.state_dict())

        elif (
            model_name == "CBM-backbone-resnet"
        ):  # Default to resnet backbone TODO adapt to vitB backbone and others
            model.load_state_dict(
                torch.load(
                    "/models_pkl/best_CBM-backbone-resnet_{}.pth".format(metadata_dataset["name"])
                ,weights_only=False).state_dict()
            )

        elif model_name == "XNES-backbone-faster-rcnn":
            model.load_state_dict(
                torch.load(
                    "/models_pkl/best_XNES-backbone-faster-rcnn_{}.pth".format(
                        metadata_dataset["name"]
                    )
                ,weights_only=False).state_dict()
            )

        elif (
            model_name == "CBM-classifier-linear" or model_name == "CBM-classifier-logistic"
        ):  # Default to resnet backbone TODO adapt to vitB backbone and others
            if pytorch_mode:
                # print parameters of the first_layer
                model.load_state_dict(
                    torch.load(
                        "/models_pkl/best_{}_{}.pth".format(model_name, metadata_dataset["name"])
                    ,weights_only=False).state_dict()
                )

                # Dataset
                dataloader_train, _, _ = data.full_dataloader_importer(
                    metadata_dataset["name"],
                    "concepts_infer_float",
                    "labels",
                    device=device,
                    training_method="sklearn",
                )

                model.scores_train = dataloader_train.List_inputs

            else:
                model.backbone.load_state_dict(
                    torch.load(
                        "/models_pkl/best_CBM-backbone-resnet_{}.pth".format(
                            metadata_dataset["name"]
                        )
                    ,weights_only=False).state_dict()
                )
                train_sklearn(metadata_dataset["name"], model, device)

        elif model_name == "XNES-classifier-logistic":
            if pytorch_mode:
                # print parameters of the first_layer
                model.load_state_dict(
                    torch.load(
                        "/models_pkl/best_{}_{}.pth".format(model_name, metadata_dataset["name"])
                    ,weights_only=False).state_dict()
                )

            else:
                model.backbone.load_state_dict(
                    torch.load(
                        "/models_pkl/best_XNES-backbone-faster-rcnn_{}.pth".format(
                            metadata_dataset["name"]
                        )
                    ,weights_only=False).state_dict()
                )
                train_sklearn(metadata_dataset["name"], model, device)
        elif model_name == "CLIP-QDA":
            if pytorch_mode:
                # print parameters of the first_layer
                model.load_state_dict(
                    torch.load(
                        "/models_pkl/best_{}_{}.pth".format(model_name, metadata_dataset["name"])
                    ,weights_only=False).state_dict()
                )
                # Dataset
                dataloader_train, _, _ = data.full_dataloader_importer(
                    metadata_dataset["name"],
                    "clip_embed_image",
                    "labels",
                    device=device,
                    training_method="sklearn",
                )
                model.clip_scores_train = model.preprocess_clip_scores(dataloader_train.List_inputs)
            else:
                # Train the model using sklearn (Loading/Saving process is not implemented given that the model is not long to train)
                train_sklearn(metadata_dataset["name"], model, device)

        elif model_name == "CLIP-zero-shot":
            pass

        else:
            raise ValueError("Network not implemented !")

    return model

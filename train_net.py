import argparse

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import data
import models
import utils


def train_dnn_loop(dataloader_train, dataloader_val, model, loss_fct, optimizer, n_epochs=50):
    """Training loop for a DNN.

    Args:
        dataloader_train (DataLoader): Dataloader for the training set.
        dataloader_val (DataLoader): Dataloader for the validation set.
        model (nn.Module): The DNN model.
        loss_fct (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        n_epochs (int, optional): The number of epochs. Defaults to 50.

    Returns:
        tuple: The best epoch and the best score.
    """
    # Constants
    best_score = 0
    best_epoch = 0

    # Training loop
    for epoch in range(n_epochs):
        for item in tqdm(dataloader_train):
            optimizer.zero_grad()
            target = torch.tensor(item["output"])
            preds = model(item["input"])
            loss = loss_fct(preds, target)
            loss.backward()
            optimizer.step()
        print(f"End of epoch {epoch} - {loss.item()}")
        if epoch % 5 == 0:
            print("Evaluating ...")
            score = utils.evaluate_(model, dataloader_val)
            print(f"Score val set: {score:.4%}")
            if score > best_score or dataloader_train.name == "pascalpart":
                best_epoch = epoch
                best_score = score

                if model.name == "resnet50-bcos" or model.name == "vitB-bcos":
                    torch.save(
                        model.state_dict(),
                        f"models_pkl/best_{model.name}_{dataloader_train.name}.pth",
                    )

                else:
                    torch.save(
                        model,
                        f"models_pkl/best_{model.name}_{dataloader_train.name}_.pth",
                    )
    return best_epoch, best_score

def train_dnn_loop_faster_rcnn(dataloader_train, dataloader_val, model, optimizer, n_epochs=50):
    """Training loop for a DNN.

    Args:
        dataloader_train (DataLoader): Dataloader for the training set.
        dataloader_val (DataLoader): Dataloader for the validation set.
        model (nn.Module): The DNN model.
        optimizer (torch.optim.Optimizer): The optimizer.
        n_epochs (int, optional): The number of epochs. Defaults to 50.

    Returns:
        tuple: The best epoch and the best score.
    """
    # Constants
    best_score = 0
    best_epoch = 0

    # Training loop
    for epoch in range(n_epochs):
        for item in tqdm(dataloader_train):
            optimizer.zero_grad()
            target = [
                {
                    "boxes": item["bbox"],
                    "labels": item["labels"],
                }
                for item in item["output"]
            ]
            losses = model(item["input"], target)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
        print(f"End of epoch {epoch} - {loss.item()}")
        if epoch % 5 == 0:
            print("Evaluating ...")
            score = utils.evaluate_fasterrcnn(model, dataloader_val)
            print(score)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    model,
                    f"models_pkl/best_{model.name}_{dataloader_train.name}.pth",
                )

    return best_epoch, best_score


def train_dnn(dataset_name: str, model: nn.Module) -> None:
    """Trains a DNN model on a given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        model (nn.Module): The DNN model to be trained.

    Raises:
        Exception: If the network is not implemented.

    Returns:
        None
    """
    lr = 0.0001
    optimizer = None
    loss_fct = None
    input_data = "images"
    output_data = "labels"
    obj_detection_mode = False
    bs_train = 128

    print(f"Training {model.name} on {dataset_name} ...")

    # Initialize the optimizer and the loss function
    if model.name == "resnet50" or model.name == "resnet50-bcos":
        optimizer = optim.Adam(
            params=[
                {"params": model.resnet50.parameters(), "lr": 0.1 * lr},
                {"params": model.linear.parameters(), "lr": lr},
            ]
        )
        loss_fct = nn.CrossEntropyLoss()

    elif model.name == "vitB" or model.name == "vitB-bcos":
        optimizer = optim.Adam(
            params=[
                {"params": model.vit_b_16.parameters(), "lr": 0.1 * lr},
                {"params": model.linear.parameters(), "lr": lr},
            ]
        )
        if model.name == "vitB-bcos":
            bs_train = 64
        loss_fct = nn.CrossEntropyLoss()
    elif model.name == "CLIP-LaBo":
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0)
        loss_fct = nn.CrossEntropyLoss()
        input_data = "clip_embed_image"
    elif model.name == "CLIP-Linear":
        optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0001)
        loss_fct = nn.CrossEntropyLoss()
        input_data = "clip_embed_image"
    elif model.name == "CBM-backbone-resnet":
        optimizer = optim.Adam(
            params=[
                {"params": model.resnet50.parameters(), "lr": 0.0001},
                {"params": model.linear.parameters(), "lr": 0.001},
            ]
        )
        loss_fct = nn.BCEWithLogitsLoss()
        output_data = "concepts"
    elif model.name == "XNES-backbone-faster-rcnn":
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
        output_data = "concepts_and_bboxes"
        obj_detection_mode = True
        bs_train = 16
    else:
        raise ValueError("Network not implemented !")

    # Load the dataloaders for the training, validation, and test sets
    dataset_train, dataset_val, dataset_test = data.full_dataloader_importer(
        dataset_name,
        input_data,
        output_data,
        device,
        training_method="pytorch",
        bs_train=bs_train,
        obj_detection_mode=obj_detection_mode,
    )

    # Train the DNN model
    if model.name == "XNES-backbone-faster-rcnn":
        best_epoch, _ = train_dnn_loop_faster_rcnn(
            dataset_train, dataset_val, model, optimizer, n_epochs=100
        )
    else:
        best_epoch, _ = train_dnn_loop(dataset_train, dataset_val, model, loss_fct, optimizer)

    # Evaluate the best checkpoint on the test set
    if model.name == "vitB-bcos" or model.name == "resnet50-bcos":
        metadata = data.metadata_importer(dataset_name)
        model = models.model_importer(model.name, metadata, device)
        model.load_state_dict(torch.load(f"models_pkl/best_{model.name}_{dataset_train.name}.pth"))

    else:
        model = torch.load(f"models_pkl/best_{model.name}_{dataset_train.name}.pth")

    if model.name == "XNES-backbone-faster-rcnn":
        score = utils.evaluate_fasterrcnn(model, dataset_test)
    else:
        score = utils.evaluate_(model, dataset_test)

    # Print the score of the best checkpoint and the best epoch
    print(
        "End. Score best checkpoint:",
        score,
        len(dataset_test),
        "Best epoch:",
        best_epoch,
    )


def test_dnn(dataset_name: str, model: nn.Module, device) -> None:
    """Tests a DNN model on a given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        model (nn.Module): The DNN model to be tested.
        device (str): The device to be used for training and evaluation.

    Returns:
        None
    """
    if model.name == "CLIP-LaBo" or model.name == "CLIP-Linear":
        input_data = "clip_embed_image"
        output_data = "labels"
        obj_detection_mode = False
    elif model.name == "CBM-backbone-resnet":
        input_data = "images"
        output_data = "concepts"
        obj_detection_mode = False
    elif model.name == "resnet50" or model.name == "vitB" or model.name == "CLIP-zero-shot":
        input_data = "images"
        output_data = "labels"
        obj_detection_mode = False
    elif model.name == "XNES-backbone-faster-rcnn":
        input_data = "images"
        output_data = "concepts_and_bboxes"
        obj_detection_mode = True
    else:
        raise ValueError("Network not implemented !")

    _, _, dataset_test = data.full_dataloader_importer(
        dataset_name,
        input_data,
        output_data,
        device,
        obj_detection_mode=obj_detection_mode,
        shuffle=False,
    )

    if model.name == "XNES-backbone-faster-rcnn":
        score = utils.evaluate_fasterrcnn(model, dataset_test)
        print(f"Score test set: {score:.4%}")

    else:
        score = utils.evaluate_(model, dataset_test)
        print(f"Score test set: {score:.4%}")


def train_sklearn(dataset_name, model, device):
    """Train a model with sklearn fit."""
    # Params train
    if model.name == "CLIP-QDA":
        input_data = "clip_embed_image"

    elif model.name == "CBM-classifier-linear" or model.name == "CBM-classifier-logistic":
        input_data = "concepts_infer_float"
        state_dict = torch.load(f"models_pkl/best_CBM-backbone-resnet_{dataset_name}.pth")
        # Iterate over state_dict keys to replace old class name with new class name
        new_state_dict = {}
        for key in state_dict.state_dict():
            new_key = key.replace("backbone.", "backbone.")
            new_state_dict[new_key] = state_dict.state_dict()[key]

        model.backbone.load_state_dict(new_state_dict)

    elif model.name == "XNES-classifier-logistic":
        input_data = "concepts_infer_float_xnesyl"
        state_dict = torch.load(f"models_pkl/best_XNES-backbone-faster-rcnn_{dataset_name}.pth")

        """# Iterate over state_dict keys to replace old class name with new class name
        new_state_dict = {}
        for key in state_dict.state_dict().keys():
            new_key = key.replace('backbone.', 'backbone.')
            new_state_dict[new_key] = state_dict.state_dict()[key]

        model.backbone.load_state_dict(new_state_dict)"""

    # Dataset
    if model.name == "PASTA":
        dataloader_train, _, dataloader_test = data.pasta_dataloader_importer(dataset_name)
    else:
        dataloader_train, _, dataloader_test = data.full_dataloader_importer(
            dataset_name, input_data, "labels", device=device, training_method="sklearn"
        )

    x_train = dataloader_train.List_inputs
    x_test = dataloader_test.List_inputs
    y_train = dataloader_train.List_outputs
    y_test = dataloader_test.List_outputs

    """print('train_train',x_train[-1])
    print('train_test',x_test[-1])"""

    # Train
    if (
        model.name == "XNES-classifier-logistic"
        or model.name == "CBM-classifier-logistic"
        or model.name == "CLIP-QDA"
    ):
        model.train_classifier(x_train, y_train)
    else:
        model.train(x_train, y_train)

    # Score
    score = model.test(x_test, y_test)

    print(f"Score sklearn test set: {score:.4%}", len(dataloader_test))

    # save model
    torch.save(model, f"models_pkl/best_{model.name}_{dataset_name}_.pth")
    print("model saved", f"models_pkl/best_{model.name}_{dataset_name}_.pth")

    return model


def train_sklearn_avg_multi_runs(
    dataset_name,
    model,
    n_runs=20,
    type_task="classification",
    input_data="CLIP_image_blur",
    device="",
    type_expl="saliency",
    sub_dataset="all",
    sigma_noise="",
    question_id="Q1",
    label_type="mean",
    restrict_split=False,
    param_cbm=False,
    file_annotations="human_annotations.json",
):
    """Same than train_sklearn but compute the average score of n_runs runs. Only implemented for the toy example of PASTA."""
    l_mse = []
    l_p_acc = []
    l_r2 = []
    l_qwk = []
    l_spear_coef = []

    for i in range(n_runs):
        
        seed = i
        # Dataset
        torch.manual_seed(seed)

        dataloader_train, dataloader_val, dataloader_test = data.pasta_dataloader_importer(
            dataset_name,
            import_criterions=input_data,
            seed=seed,
            type_task=type_task,
            device=device,
            type_expl=type_expl,
            sub_dataset=sub_dataset,
            sigma_noise=sigma_noise,
            question_id=question_id,
            label_type=label_type,
            restrict_split=restrict_split,
            position_oracle=i,
            param_cbm=param_cbm,
            file_annotations=file_annotations,
        )

        x_train = dataloader_train.List_inputs
        x_val = dataloader_val.List_inputs
        x_test = dataloader_test.List_inputs

        y_train = dataloader_train.List_outputs
        y_val = dataloader_val.List_outputs
        y_test = dataloader_test.List_outputs

        if (
            label_type == "test_oracle"
            or label_type == "test_random"
            or label_type == "test_global_mean"
        ):
            y_test_oracle = dataloader_test.labels_oracle
            d_results = model.test(y_test_oracle, y_test, given_label=True)

        else:
            # Train
            model.train(
                x_train,
                y_train,
                data_in_val=x_val,
                data_out_val=y_val,
                data_in_test=x_test,
                data_out_test=y_test,
            )

            # Score
            d_results = model.test(x_test, y_test)

            if model.name == "PASTA-pytorch":
                # Save state dict
                torch.save(model.best_model_state, f"models_pkl/PASTA_{question_id}.pth")
                for layer in model.classifier.net.children():
                    if hasattr(layer, "reset_parameters"):
                        print("Resetting parameters")
                        layer.reset_parameters()

        l_p_acc.append(d_results["Pseudo-Classification Accuracy"])
        l_mse.append(d_results["MSE"])
        l_r2.append(d_results["R2"])
        l_qwk.append(d_results["Cohen's Kappa"])
        l_spear_coef.append(d_results["Spearman Coef"])

    l_spear_coef = [seed.statistic for seed in l_spear_coef]

    print(f"Sanity check: Score Spearman Coef test set: {l_spear_coef}")
    print(f"Sanity check: Score Cohen's Kappa test set: {l_qwk}")
    print(f"Sanity check: Score R2 test set: {l_r2}")
    print(f"Sanity check: Score MSE test set: {l_mse}")
    print(f"Score pseudo acc test set: {np.mean(l_p_acc):.4%}")
    print(f"Score r2 test set: {np.mean(l_r2):.4%}")
    print(f"Score mse test set: {np.mean(l_mse):.4%}")
    print(f"Score qwk test set: {np.mean(l_qwk):.4%}")
    # Remove nan values from l_spear_coef
    l_qwk = [qwk for qwk in l_qwk if qwk != 0.0]
    l_mse = [mse for mse in l_mse if mse != 0.0]
    l_spear_coef = [coef for coef in l_spear_coef if not np.isnan(coef)]
    print(f"Score spearman test set: {np.mean(l_spear_coef):.4%}")

    return {
        "MSE": {"mean": np.mean(l_mse), "var": np.std(l_mse)},
        "Pseudo-Classification Accuracy": {"mean": np.mean(l_p_acc), "var": np.std(l_p_acc)},
        "R2": {"mean": np.mean(l_r2), "var": np.std(l_r2)},
        "Cohen's Kappa": {"mean": np.mean(l_qwk), "var": np.std(l_qwk)},
        "Spearman Coef": {"mean": np.mean(l_spear_coef), "var": np.std(l_spear_coef)},
    }


def test_sklearn(dataset_name, model, device):
    """Test a model with sklearn fit."""
    # Params test
    if model.name == "CLIP-QDA":
        input_data = "clip_embed_image"

    elif model.name == "CBM-classifier-linear" or model.name == "CBM-classifier-logistic":
        input_data = "concepts_infer_float"

    elif model.name == "XNES-classifier-logistic":
        input_data = "concepts_infer_float_xnesyl"

    # Dataset
    _, _, dataloader_test = data.full_dataloader_importer(
        dataset_name, input_data, "labels", device=device, training_method="sklearn"
    )

    x_test = dataloader_test.List_inputs
    y_test = dataloader_test.List_outputs

    """print('test',x_test[-1])"""

    # Score
    score = model.test(x_test, y_test)

    """torch.save(
        model,
        f"models_pkl/best_{model.name}_{dataloader_test.name}_.pth",
    )
    print('saved!')"""

    print(f"Score test set: {score:.4%}")


## Close writer

if __name__ == "__main__":
    ## Parser
    def parse_args():
        # Training settings
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "--network",
            type=str,
            default="resnet50",
            help="network (resnet50, vitB, CLIP-zero-shot, CLIP-QDA, CLIP-LaBo, CLIP-Linear, CBM-backbone-resnet, CBM-classifier-linear, CBM-classifier-logistic, XNES-backbone-faster-rcnn, XNES-classifier-logistic)",
        )
        parser.add_argument("--gpus", type=str, default="0", help="gpu ids")  # TODO multigpu
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="pascalpart",
            help="set of input data to use in [pascalpart,monumai,coco,catsdogscars]",
        )
        parser.add_argument(
            "--test_only", action="store_true", help="only test the model on the test set"
        )
        return parser.parse_args()

    # Args
    args = parse_args()

    ## Others parameters
    device = f"cuda:{args.gpus}" if (torch.cuda.is_available() and args.gpus != "-1") else "cpu"

    if args.test_only:
        print("Testing ...")
        if (
            args.network == "CLIP-QDA"
            or args.network == "CLIP-QDA-sample"
            or args.network == "XNES-classifier-logistic"
        ):
            metadata = data.metadata_importer(args.dataset_name)
            model = models.model_importer(args.network, metadata, device, load_model=True)
            test_sklearn(args.dataset_name, model, device)

        elif (
            args.network == "resnet50"
            or args.network == "resnet50-bcos"
            or args.network == "vitB"
            or args.network == "vitB-bcos"
            or args.network == "CLIP-LaBo"
            or args.network == "CBM-backbone-resnet"
            or args.network == "CLIP-Linear"
            or args.network == "CLIP-zero-shot"
            or args.network == "XNES-backbone-faster-rcnn"
        ):
            metadata = data.metadata_importer(args.dataset_name)
            model = models.model_importer(args.network, metadata, device, load_model=True)
            test_dnn(args.dataset_name, model, device)

    else:
        ## Train loop if DNN
        print("Training ...")

        metadata = data.metadata_importer(args.dataset_name)

        ## Import model
        model = models.model_importer(args.network, metadata, device)

        ## Train
        if (
            args.network == "CLIP-QDA"
            or args.network == "CBM-classifier-linear"
            or args.network == "CBM-classifier-logistic"
            or args.network == "XNES-classifier-logistic"
        ):
            train_sklearn(args.dataset_name, model, device)

        elif (
            args.network == "resnet50"
            or args.network == "resnet50-bcos"
            or args.network == "vitB"
            or args.network == "vitB-bcos"
            or args.network == "CLIP-LaBo"
            or args.network == "CBM-backbone-resnet"
            or args.network == "CLIP-Linear"
            or args.network == "XNES-backbone-faster-rcnn"
        ):
            train_dnn(args.dataset_name, model)

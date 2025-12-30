import argparse
from pathlib import Path

import pandas as pd
import torch

import models
from train_net import train_sklearn_avg_multi_runs

if __name__ == "__main__":
    ## Parser
    def parse_args():
        # Training settings
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "--network",
            type=str,
            default="PASTA-pytorch",
            help="network in [PASTA-logistic,PASTA-linear,PASTA-QDA,PASTA-svm,PASTA-elastic,PASTA-ridge,PASTA-lasso,PASTA-pytorch,PASTA-mlp]",
        )
        parser.add_argument("--gpus", type=str, default="0", help="gpu ids")  # TODO multigpu
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="true_sklearn",
            help="set of input data to use in [toy_sklearn, true_sklearn]",
        )
        parser.add_argument(
            "--sub_dataset_name",
            type=str,
            default="all",
            help="subdataset to use in [coco, pascalpart, monumai, catsdogscars, all]",
        )
        parser.add_argument(
            "--type_task",
            type=str,
            default="regression",
            help="type of task to use in [classification, regression]",
        )
        parser.add_argument(
            "--input_data",
            type=str,
            default="SIGLIP_CBM_text+SIGLIP_heatmap+labels",
            help="what to consider as input in [metrics,CLIP_image_blur,CLIP_heatmap,CBM_activations,CLIP_CBM_text,LLaVa_CBM_text+LLaVa_heatmap,BLIP_CBM_text+BLIP_heatmap,CLIP_CBM_text+CLIP_heatmap,CLIP_CBM_text+CLIP_heatmap+labels,SIGLIP_CBM_text+SIGLIP_heatmap+labels]",
        )
        parser.add_argument(
            "--type_expl",
            type=str,
            default="all",
            help="type of explanation in [saliency,cbm,all]",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.001,
            help="if ridge or lasso, change the parameter alpha, for pytorch, learning rate",
        )
        parser.add_argument(
            "--sigma_noise",
            type=str,
            default="05",
            help="noise in the input labels, in 05 (sigma=0.5),1 (sigma=1),15 (sigma=1.5)]",
        )
        parser.add_argument(
            "--id_question",
            type=str,
            default="Q1",
            help="For the real dataset, the id of the question in [Q1,Q2,Q3,Q4,Q5,Q6]",
        )
        parser.add_argument(
            "--label_type",
            type=str,
            default="mode",
            help="how to compute the label from the annotator ratings in [mean,median,mode,test_oracle,test_random,test_global_mean]",
        )
        parser.add_argument(
            "--restrict_split",
            type=str,
            default="img_and_xai_id",
            help="if img_id, avoid to have the same image in several splits, if xai_id avoid to have the same explanation in several splits",
        )
        parser.add_argument(
            "--param_cbm",
            type=int,
            default=15,
            help="custom param for abaltaion studies",
        )
        parser.add_argument(
            "--test_only", action="store_true", help="only test the model on the test set"
        )
        return parser.parse_args()

    # Args
    args = parse_args()

    ## Load criterions

    ## Others parameters
    device = f"cuda:{args.gpus}" if (torch.cuda.is_available() and args.gpus != "-1") else "cpu"

    # imput shape
    if args.input_data == "metrics":
        input_size = 9

    elif (
        ("CLIP" in args.input_data)
        and ("label" not in args.input_data)
        and ("dataset" not in args.input_data)
        or ("BLIP" in args.input_data)
        and ("label" not in args.input_data)
        and ("dataset" not in args.input_data)
    ):
        input_size = 768

    elif (
        ("LLaVa" in args.input_data)
        and ("label" not in args.input_data)
        and ("dataset" not in args.input_data)
    ):
        input_size = 4096

    elif (
        ("SIGLIP" in args.input_data)
        and ("label" not in args.input_data)
        and ("dataset" not in args.input_data)
    ):
        input_size = 1024

    elif "dataset" and "SIGLIP" in args.input_data and ("label" not in args.input_data):
        input_size = 1024 + 4

    elif "label" and "SIGLIP" in args.input_data:
        input_size = 1024 + 26

    elif (
        "label"
        and "CLIP" in args.input_data
        or "label"
        and "BLIP" in args.input_data
        or "label"
        and "EVA02" in args.input_data
    ):
        input_size = 768 + 26

    else:
        input_size = 0
        print("Input data not recognized, please check the input_data argument.")

    # Bad code to rule pytorch params

    dict_params = {
        "batch_size": 256,
        "epochs": 600,
        "alpha": 10,
        "beta": 0.01,
        "gamma": 0.01,
        "lr": 0.000002,
        "w_decay": 1e-6,
        "optimizer": "Adam",
        "template": "",
        "N_top": args.param_cbm,
        "hidden_size": [512, 64],
        "file_annotations": "human_annotations.json",
    }

    ## Import model
    model = models.model_importer(
        args.network,
        None,
        device,
        type_task=args.type_task,
        alpha=args.alpha,
        input_size=input_size,
        dict_params=dict_params,
    )

    ## Train
    d_results = train_sklearn_avg_multi_runs(
        args.dataset_name,
        model,
        type_task=args.type_task,
        input_data=args.input_data,
        device=device,
        n_runs=1,
        type_expl=args.type_expl,
        sub_dataset=args.sub_dataset_name,
        sigma_noise=args.sigma_noise,
        question_id=args.id_question,
        label_type=args.label_type,
        restrict_split=args.restrict_split,
        param_cbm=args.param_cbm,
        file_annotations=dict_params["file_annotations"],
    )

    # Filename for the results table
    filename = f"results_pasta2_{args.type_expl}.csv"

    m_p_acc = d_results["Pseudo-Classification Accuracy"]["mean"]
    v_p_acc = d_results["Pseudo-Classification Accuracy"]["var"]

    m_mse = d_results["MSE"]["mean"] * 16
    v_mse = d_results["MSE"]["var"] * 16

    m_r2 = d_results["R2"]["mean"]
    v_r2 = d_results["R2"]["var"]

    m_qwk = d_results["Cohen's Kappa"]["mean"]
    v_qwk = d_results["Cohen's Kappa"]["var"]

    m_spear_coef = d_results["Spearman Coef"]["mean"]
    v_spear_coef = d_results["Spearman Coef"]["var"]

    if args.sigma_noise == "05":
        sigma_print = "0.5"

    elif args.sigma_noise == "1":
        sigma_print = "1"

    elif args.sigma_noise == "15":
        sigma_print = "1.5"

    # Create a dictionary to hold the new results

    new_entry = {
        "Regressor": f"{args.network}_({args.input_data})",
        "ID question": f"{args.id_question}",
        "label_type": f"{args.label_type}",
        "restrict_split": f"{args.restrict_split}",
        "custom_param": f"{args.param_cbm}",
        "Alpha": f"{args.alpha}",
        "MSE": f"{m_mse:.4} +- {v_mse:.4}",
        "Cohen's Kappa": f"{m_qwk:.4} +- {v_qwk:.4}",
        "Spearman Coef": f"{m_spear_coef:.4} +- {v_spear_coef:.4}",
        "Dict_params": f"batch_size={dict_params['batch_size']}, epochs={dict_params['epochs']}, alpha={dict_params['alpha']}, beta={dict_params['beta']}, gamma={dict_params['gamma']}, lr={dict_params['lr']}, w_decay={dict_params['w_decay']}, optimizer={dict_params['optimizer']}, template={dict_params['template']}, N_top={dict_params['N_top']}, hidden_size={dict_params['hidden_size']}",
    }

    # Check if the file exists
    if Path(filename).exists():
        # Load existing data
        df = pd.read_csv(filename)
        # Append the new entry
        df = df._append(new_entry, ignore_index=True)
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame([new_entry])

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

    print(f"Results saved to {filename}")

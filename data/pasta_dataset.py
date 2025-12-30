import json
import random
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import open_clip
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm

import data
import utils
from BLIP.models.blip import blip_feature_extractor


class PASTAdataset(torch.utils.data.Dataset):
    """The true annotated dataset for PASTA."""

    def __init__(
        self,
        phase="train",
        root_ann="/media/remi/RemiPro/PASTA_dataset_w_activations/",
        root="/media/remi/RemiPro/PASTA_dataset_w_activations/",
        import_criterions=False,
        seed=493,
        type_task="regression",
        type_expl="saliency",
        dataset="catsdogscars",
        device="",
        question_id="Q1",
        label_type="mean",
        restrict_split=False,
        position_oracle=0,
        param_cbm=False,
        file_annotations="human_annotations.json",
        callibrate_mode=False,
        fixed_set_test=False,
    ):
        """Initialize the SampleDatasetCatsDogsCars dataset.

        Args:
            phase (str, optional): The phase of the dataset. Defaults to 'train'.
            root (str, optional): The root directory of the dataset. Defaults to 'data_samples/'.
            import_criterions (str, optional): Whether to import the criterions. Defaults to False.
            seed (int, optional): The seed of the dataset shuffle. Defaults to 493.
            type_task (str, optional): The type of the task. Defaults to 'classification'.
            type_expl (str, optional): The type of explanation to deal with in [saliency,cbm]. Defaults to "saliency".
            dataset (str, optional): The dataset to use. Defaults to 'catsdogscars'.
            device (str, optional): The device to use. Defaults to "".
            question_id (str, optional): The question id to use. Defaults to 'Q1'.
            label_type (str, optional): The type of label. Defaults to "mean".
            restrict_split (bool, optional): Whether to restrict the split. Defaults to False.
            position_oracle (int, optional): The position of the oracle. Defaults to 0.
            param_cbm (bool, optional): Whether to use the param cbm. Defaults to False.
            callibrate_mode (bool, optional): Whether to callibrate between 1st and 2nd batch. Defaults to False.
            file_annotations (str, optional): The file containing the human annotations. Defaults to "human_annotations.json".
            fixed_set_test (bool, optional): Whether to use a fixed set for testing. Defaults to False.
        """
        # init moche, TODO refaire en plus propre

        super().__init__()

        if callibrate_mode:
            print("!!! WARNING: callibrate_mode is set to True !!!")

        self.root_ann = root_ann
        self.root = root
        self.List_img_pth = []
        self.List_exp_pth = []
        self.List_act_pth = []
        self.List_gtclass = []
        self.List_predictedclass = []
        self.List_label = []
        self.List_xai_id = []
        self.List_img_id = []
        self.List_criterions = []
        self.import_criterions = import_criterions
        self.type_expl = type_expl
        self.device = device

        flag_saved = False

        if root == "result_dataset_1st_batch/":
            print("!!! WARNING: root is set to result_dataset_1st_batch/ !!!")

        if (
            label_type == "test_oracle"
            or label_type == "test_random"
            or label_type == "test_global_mean"
        ):
            self.labels_oracle = []

        if self.import_criterions:
            self.List_inputs = []
            self.List_outputs = []

        # Bad code to exclude criterions
        select_criterions = [
            "complexity_10",
            "complexity_20",
            "complexity_30",
            "classification_logistic",
            "classification_qda",
            "classification_svm",
            "variance_gauss",
            "variance_sharp",
            "variance_bright",
        ]

        if "SIGLIP" in self.import_criterions:
            # Load SigLIP model
            model_name = "ViT-L-16-SigLIP-384"
            pretrained = "webli"
            self.openclip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            self.openclip_model.to(self.device)
            self.openclip_model.eval()
            self.model_type = "SIGLIP"
            if "SIGLIP_weighted_sum" in self.import_criterions:
                d_clips_embed_concepts = {}
                for dataset_name in ["pascalpart", "catsdogscars", "coco", "monumai"]:
                    metadata = data.metadata_importer(dataset_name)
                    list_concepts = metadata["labeled_concepts"]
                    tokens = tokenizer(list_concepts).to(self.device)
                    d_clips_embed_concepts[dataset_name] = (
                        self.openclip_model.encode_text(tokens).cpu().detach().numpy()
                    )

        elif "EVA02" in self.import_criterions:
            # Load SigLIP model
            model_name = "EVA02-L-14-336"
            pretrained = "merged2b_s6b_b61k"
            self.openclip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            self.openclip_model.to(self.device)
            self.openclip_model.eval()
            self.model_type = "EVA02"
            if "EVA02_weighted_sum" in self.import_criterions:
                d_clips_embed_concepts = {}
                for dataset_name in ["pascalpart", "catsdogscars", "coco", "monumai"]:
                    metadata = data.metadata_importer(dataset_name)
                    list_concepts = metadata["labeled_concepts"]
                    tokens = tokenizer(list_concepts).to(self.device)
                    d_clips_embed_concepts[dataset_name] = (
                        self.openclip_model.encode_text(tokens).cpu().detach().numpy()
                    )

        elif "CLIP" in self.import_criterions:
            # Load CLIP model
            self.clip_net, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
            self.clip_net.eval()
            if "CLIP_weighted_sum" in self.import_criterions:
                d_clips_embed_concepts = {}
                for dataset_name in ["pascalpart", "catsdogscars", "coco", "monumai"]:
                    metadata = data.metadata_importer(dataset_name)
                    list_concepts = metadata["labeled_concepts"]
                    tokens = clip.tokenize(list_concepts).to(self.device)
                    d_clips_embed_concepts[dataset_name] = (
                        self.clip_net.encode_text(tokens).cpu().detach().numpy()
                    )

        elif "LLaVa" in self.import_criterions:
            model_path = "ggml-model-q5_k.gguf"
            clip_model_path = "mmproj-model-f16.gguf"
            self.llava_net = utils.LLaVa_utils(
                model_path=model_path, clip_model_path=clip_model_path
            )

        elif "BLIP" in self.import_criterions:
            model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"
            self.blip_net = (
                blip_feature_extractor(pretrained=model_url, image_size=224, vit="base")
                .eval()
                .to(self.device)
            )

        if type_task == "classification":
            with Path(root + "anns_toy_PASTA.json").open("r") as fp:
                list_ann_samples = json.load(fp)

        elif type_task == "regression":
            # Version Gianni
            with Path(root_ann + "/" + file_annotations).open("r") as fp:
                list_ann_samples = json.load(fp)
                if file_annotations == "human_annotations_just_2nd_batch.json":
                    print("!!!! WARNING: just 2nd batch")
                elif file_annotations == "human_annotations_just_1st_batch.json":
                    print("!!!! WARNING: just 1st batch")

            if dataset == "not_coco":
                print("Filtering the dataset to only include images that are not in COCO")
                list_ann_samples = [x for x in list_ann_samples if "coco" not in x["img"]]

            elif dataset != "all":
                print("Filtering the dataset to only include images from:", dataset)
                list_ann_samples = [
                    x for x in list_ann_samples if x["img"].split("/")[-1].split("_")[0] == dataset
                ]

            if type_expl == "saliency":
                list_ann_samples = [
                    x
                    for x in list_ann_samples
                    if x["explication"].split("/")[-1].split("_")[2]
                    not in ["39", "40", "41", "42", "43", "44", "45", "48", "49"]
                ]
            elif type_expl == "cbm":
                list_ann_samples = [
                    x
                    for x in list_ann_samples
                    if x["explication"].split("/")[-1].split("_")[2]
                    in ["39", "40", "41", "42", "43", "44", "45", "48", "49"]
                ]

            # Shuffle the image_ids randomly
            np.random.seed(seed)
            np.random.shuffle(list_ann_samples)

            # Group the data entries by image_id
            grouped_data = defaultdict(list)

            if True:
                set_img_ids = set()
                set_xai_ids = set()

            for i_entry, entry in enumerate(list_ann_samples):
                # Extract the image_id from the explication path
                if restrict_split == "img_id":
                    id_to_keep = entry["explication"].split("/")[-1].split("_")[1]
                elif restrict_split == "xai_id":
                    id_to_keep = entry["explication"].split("/")[-1].split("_")[2]
                elif restrict_split == "img_and_xai_id":
                    id_to_keep = (
                        entry["explication"].split("/")[-1].split("_")[1]
                        + "_"
                        + entry["explication"].split("/")[-1].split("_")[2]
                    )
                    set_img_ids.add(entry["explication"].split("/")[-1].split("_")[1])
                    set_xai_ids.add(entry["explication"].split("/")[-1].split("_")[2])
                else:
                    id_to_keep = i_entry  # Bad but easy to code

                # Group by image_id
                grouped_data[id_to_keep].append(entry)

            # Get all unique image_ids
            all_image_ids = list(grouped_data.keys())

            # Determine split sizes (e.g., 70% train, 15% validation, 15% test)
            train_size = int(0.7 * len(all_image_ids))
            val_size = int(0.15 * len(all_image_ids))

            # Split the image_ids into train, validation, and test sets
            if restrict_split == "img_and_xai_id":
                train_size_img_id = int(0.7 * len(set_img_ids))
                train_size_xai_id = int(0.7 * len(set_xai_ids))
                l_img_id = sorted(set_img_ids)
                l_xai_id = sorted(set_xai_ids)
                np.random.shuffle(l_img_id)
                np.random.shuffle(l_xai_id)
                valid_train_img_ids = l_img_id[:train_size_img_id]
                valid_train_xai_ids = l_xai_id[:train_size_xai_id]
                train_image_ids = []
                valtest_image_ids = []

                id_img_test = l_img_id[train_size_img_id:]
                L_img_ok = []
                for sample in list_ann_samples:
                    if (sample["img"].split('/')[-1].split('_')[1] in id_img_test) :
                        if not (sample["img"].split('/')[-1] in L_img_ok) : 
                            L_img_ok.append(sample["img"].split('/')[-1])

                for index in all_image_ids:
                    img_id = index.split("_")[0]
                    xai_id = index.split("_")[1]
                    if (img_id in valid_train_img_ids) and (xai_id in valid_train_xai_ids):
                        train_image_ids.append(index)
                    elif (img_id not in valid_train_img_ids) and (
                        xai_id not in valid_train_xai_ids
                    ):
                        valtest_image_ids.append(index)
                val_image_ids = valtest_image_ids[: int(0.5 * len(valtest_image_ids))]
                test_image_ids = valtest_image_ids[int(0.5 * len(valtest_image_ids)) :]
            elif restrict_split == "dataset_and_xai_id":
                ids_monumai = [f"{id_img}" for id_img in range(550, 775)] + [
                    f"{id_img}" for id_img in range(25)
                ]
                ids_catsdogscars = [f"{id_img}" for id_img in range(100, 325)] + [
                    f"{id_img}" for id_img in range(50, 75)
                ]
                ids_coco = [f"{id_img}" for id_img in range(325, 550)] + [
                    f"{id_img}" for id_img in range(75, 100)
                ]
                ids_pascalpart = [f"{id_img}" for id_img in range(775, 1000)] + [
                    f"{id_img}" for id_img in range(25, 50)
                ]
                dict_ids_dataset = {
                    "coco": ids_coco,
                    "monumai": ids_monumai,
                    "catsdogscars": ids_catsdogscars,
                    "pascalpart": ids_pascalpart,
                }
                train_size_xai_id = int(0.7 * len(set_xai_ids))
                # Select randomly 3 dataset for train, one for val and test
                selected_dataset = random.sample(list(dict_ids_dataset.keys()), 3)
                l_img_id = []
                for dataset in selected_dataset:
                    l_img_id += dict_ids_dataset[dataset]
                l_img_id = sorted(l_img_id)
                l_xai_id = sorted(set_xai_ids)
                np.random.shuffle(l_img_id)
                np.random.shuffle(l_xai_id)
                valid_train_img_ids = l_img_id
                valid_train_xai_ids = l_xai_id[:train_size_xai_id]
                train_image_ids = []
                valtest_image_ids = []
                for index in all_image_ids:
                    img_id = index.split("_")[0]
                    xai_id = index.split("_")[1]
                    if (img_id in valid_train_img_ids) and (xai_id in valid_train_xai_ids):
                        train_image_ids.append(index)
                    elif (img_id not in valid_train_img_ids) and (
                        xai_id not in valid_train_xai_ids
                    ):
                        valtest_image_ids.append(index)
                val_image_ids = valtest_image_ids[: int(0.5 * len(valtest_image_ids))]
                test_image_ids = valtest_image_ids[int(0.5 * len(valtest_image_ids)) :]
            else:
                train_image_ids = all_image_ids[:train_size]
                val_image_ids = all_image_ids[train_size : train_size + val_size]
                test_image_ids = all_image_ids[train_size + val_size :]

            if phase == "train":
                list_ann_phase = [
                    entry for img_id in train_image_ids for entry in grouped_data[img_id]
                ]

            elif phase == "val":
                list_ann_phase = [
                    entry for img_id in val_image_ids for entry in grouped_data[img_id]
                ]

            elif phase == "test":
                list_ann_phase = [
                    entry for img_id in test_image_ids for entry in grouped_data[img_id]
                ]

            elif phase == "all":
                list_ann_phase = list_ann_samples

            if restrict_split:
                np.random.shuffle(list_ann_samples)

            if self.import_criterions == "metrics":
                with Path("result_dataset/criterion_true.json").open("r") as fp:
                    self.dict_criterions = json.load(fp)
        # Version Gianni
        for entry in tqdm(list_ann_phase):
            xai_id = entry["explication"].split("/")[-1].split("_")[2]
            img_id = entry["explication"].split("/")[-1].split("_")[1]
            self.List_img_pth.append(entry["img"])
            self.List_exp_pth.append(entry["explication"])
            """self.List_act_pth.append(activations_pth)"""
            if xai_id in ["39", "40", "41", "42", "43", "44", "45", "48", "49"]:
                activations_pth = (
                    self.root + entry["explication"].replace(".png", ".json").replace("/expl/", "/activations/")
                )
            elif xai_id not in ["39", "40", "41", "42", "43", "44", "45", "48", "49"]:
                activations_pth = (
                    self.root + entry["explication"].replace(".png", ".npy").replace("/expl/", "/activations/")
                )
            self.List_act_pth.append(activations_pth)
            self.List_gtclass.append(entry["GTclass"])
            self.List_img_id.append(img_id)
            self.List_xai_id.append(xai_id)
            self.List_predictedclass.append(entry["predictedclass"])

            # Extract ratings for the given question ID (e.g., Q1)
            annotations = entry["Annotation"]
            ratings = [
                annotation[question_id] for annotation in annotations if question_id in annotation
            ]

            # Calculate the average rating for the entry
            if ratings:
                if label_type == "mean":
                    avg_rating = sum(ratings) / len(ratings)
                elif label_type == "median":
                    avg_rating = np.median(ratings)
                elif label_type == "mode":
                    avg_rating = stats.mode(ratings)[0]
                elif label_type == "test_oracle":
                    avg_rating = stats.mode(ratings)[0]
                    """avg_rating = sum(ratings) / len(ratings)"""
                    self.labels_oracle.append((ratings[position_oracle] - 1) / 4)
                elif label_type == "test_random":
                    avg_rating = stats.mode(ratings)[0]
                    self.labels_oracle.append(
                        np.random.randint(0, 4) / 4 + np.random.random() * 0.01
                    )  # Slall noise to avoid exact same values
                elif label_type == "test_global_mean":
                    avg_rating = stats.mode(ratings)[0]
                    self.labels_oracle.append(
                        (2.8568530805687202 - 1) / 4 + np.random.random() * 0.01
                    )  # Slall noise to avoid exact same values
                else:
                    ValueError("No mode available", label_type)

            else:
                ValueError("No ratings found for the entry")

            self.List_label.append(avg_rating)

            if self.import_criterions == "CLIP_image_blur":
                input_image = np.array(Image.open(self.root+entry["img"]))
                activations = np.clip(np.load(activations_pth), 0, 1)
                if xai_id in ["46", "47"]:
                    activations = activations[:, :, -1]
                image_blur = input_image * activations[:, :, np.newaxis]
                img_tensor = (
                    self.preprocess(Image.fromarray(np.uint8(image_blur)))
                    .unsqueeze(0)
                    .to(self.device)
                )
                image_embedding = (
                    self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                )
                self.List_inputs.append(image_embedding)

            if self.import_criterions == "CLIP_image_blur_treshold":
                input_image = np.array(Image.open(self.root+entry["img"]))
                activations = np.clip(np.load(activations_pth), 0, 1)
                activations = utils.modify_activations_blur(activations)
                if xai_id in ["46", "47"]:
                    activations = activations[:, :, -1]
                image_blur = input_image * activations[:, :, np.newaxis]
                img_tensor = (
                    self.preprocess(Image.fromarray(np.uint8(image_blur)))
                    .unsqueeze(0)
                    .to(self.device)
                )
                image_embedding = (
                    self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                )
                self.List_inputs.append(image_embedding)

            elif self.import_criterions == "CLIP_heatmap":
                input_image = np.array(Image.open(self.root+entry["img"]))
                explication = np.array(Image.open(self.root+entry["explication"]))
                img_tensor = (
                    self.preprocess(Image.fromarray(np.uint8(explication)))
                    .unsqueeze(0)
                    .to(self.device)
                )
                image_embedding = (
                    self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                )
                self.List_inputs.append(image_embedding)

            elif self.import_criterions == "CBM_activations":
                input_image = np.array(Image.open(self.root+entry["img"]))
                with Path(activations_pth).open("r") as fp:
                    activations_dict = json.load(fp)
                dataset_name = entry["img"].split("/")[2].split("_")[0]
                activations = utils.activations_from_dict(activations_dict, dataset_name, xai_id)
                self.List_inputs.append(activations)

            elif self.import_criterions == "CBM_CLIP_weighted_sum":
                input_image = np.array(Image.open(self.root+entry["img"]))
                with Path(activations_pth).open("r") as fp:
                    activations_dict = json.load(fp)
                dataset_name = entry["img"].split("/")[2].split("_")[0]
                activations = utils.activations_from_dict(
                    activations_dict, dataset_name, xai_id, normalize=True
                )
                weighted_embeds = np.sum(
                    np.array(
                        [
                            activations[i] * d_clips_embed_concepts[dataset_name][i]
                            for i in range(len(activations))
                        ]
                    ),
                    axis=0,
                )
                self.List_inputs.append(weighted_embeds)

            elif self.import_criterions == "CLIP_CBM_text":
                input_image = np.array(Image.open(self.root+entry["img"]))
                with Path(activations_pth).open("r") as fp:
                    activations_dict = json.load(fp)
                dataset_name = entry["img"].split("/")[2].split("_")[0]
                activations = utils.activations_from_dict(activations_dict, dataset_name, xai_id)
                text = utils.sentence_from_activations(
                    activations, dataset_name, xai_id, top_n=param_cbm
                )
                tokens = clip.tokenize(text).to(self.device)
                text_embeding = self.clip_net.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                self.List_inputs.append(text_embeding)

            elif self.import_criterions == "BLIP_CBM_text+BLIP_heatmap":
                activations = activations_pth
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    dummy_image = torch.empty((1, 3, 224, 224)).to(self.device)
                    with torch.no_grad():
                        text_feature = self.blip_net(dummy_image, text, mode="text")[0, 0]
                    self.List_inputs.append(text_feature.cpu().detach().numpy())

                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        torch.tensor(explication)
                        .unsqueeze(0)
                        .to(self.device)
                        .permute(0, 3, 1, 2)
                        .to(torch.float32)
                    )
                    with torch.no_grad():
                        img_feature = self.blip_net(img_tensor, "", mode="image")[0, 0]
                    self.List_inputs.append(img_feature.cpu().detach().numpy())

            elif self.import_criterions == "BLIP_CBM_text+BLIP_heatmap+labels":
                activations = activations_pth
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    dummy_image = torch.empty((1, 3, 224, 224)).to(self.device)
                    with torch.no_grad():
                        text_feature = self.blip_net(dummy_image, text, mode="text")[0, 0]
                    # Concat one_hot_label and text_embeding
                    self.List_inputs.append(
                        np.concatenate((one_hot_label, text_feature.cpu().detach().numpy()))
                    )

                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        torch.tensor(explication)
                        .unsqueeze(0)
                        .to(self.device)
                        .permute(0, 3, 1, 2)
                        .to(torch.float32)
                    )
                    with torch.no_grad():
                        img_feature = self.blip_net(img_tensor, "", mode="image")[0, 0]
                    self.List_inputs.append(
                        np.concatenate((one_hot_label, img_feature.cpu().detach().numpy()))
                    )

            elif self.import_criterions == "BLIP_CBM_text+BLIP_heatmap+dataset":
                activations = activations_pth
                dataset = entry["img"].split("/")[-1].split("_")[0]
                one_hot_label = utils.convert_dataset_to_one_hot(dataset)
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    dummy_image = torch.empty((1, 3, 224, 224)).to(self.device)
                    with torch.no_grad():
                        text_feature = self.blip_net(dummy_image, text, mode="text")[0, 0]
                    # Concat one_hot_label and text_embeding
                    self.List_inputs.append(
                        np.concatenate((one_hot_label, text_feature.cpu().detach().numpy()))
                    )

                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        torch.tensor(explication)
                        .unsqueeze(0)
                        .to(self.device)
                        .permute(0, 3, 1, 2)
                        .to(torch.float32)
                    )
                    with torch.no_grad():
                        img_feature = self.blip_net(img_tensor, "", mode="image")[0, 0]
                    self.List_inputs.append(
                        np.concatenate((one_hot_label, img_feature.cpu().detach().numpy()))
                    )

            elif self.import_criterions == "SIGLIP_CBM_text+SIGLIP_heatmap":
                activations = activations_pth
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = tokenizer(text).to(self.device)
                    text_embedding = (
                        self.openclip_model.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    # Concatenate one_hot_label and text_embedding
                    self.List_inputs.append(text_embedding)
                elif activations.endswith(".npy"):
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(explication)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.openclip_model.encode_image(img_tensor)
                        .cpu()
                        .detach()
                        .numpy()
                        .squeeze(0)
                    )
                    # Concatenate one_hot_label and image_embedding
                    self.List_inputs.append(image_embedding)

            elif (
                self.import_criterions == "SIGLIP_CBM_text+SIGLIP_heatmap+labels"
                or self.import_criterions == "EVA02_CBM_text+EVA02_heatmap+labels"
            ):
                activations = activations_pth
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = tokenizer(text).to(self.device)
                    text_embedding = (
                        self.openclip_model.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    # Concatenate one_hot_label and text_embedding
                    self.List_inputs.append(np.concatenate((one_hot_label, text_embedding)))
                elif activations.endswith(".npy"):
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(explication)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.openclip_model.encode_image(img_tensor)
                        .cpu()
                        .detach()
                        .numpy()
                        .squeeze(0)
                    )
                    # Concatenate one_hot_label and image_embedding
                    self.List_inputs.append(np.concatenate((one_hot_label, image_embedding)))

            elif (
                self.import_criterions == "SIGLIP_CBM_text+SIGLIP_heatmap+dataset"
                or self.import_criterions == "EVA02_CBM_text+EVA02_heatmap+dataset"
            ):
                activations = activations_pth

                dataset = entry["img"].split("/")[-1].split("_")[0]
                one_hot_label = utils.convert_dataset_to_one_hot(dataset)

                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = tokenizer(text).to(self.device)
                    text_embedding = (
                        self.openclip_model.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    # Concatenate one_hot_label and text_embedding
                    self.List_inputs.append(np.concatenate((one_hot_label, text_embedding)))
                elif activations.endswith(".npy"):
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(explication)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.openclip_model.encode_image(img_tensor)
                        .cpu()
                        .detach()
                        .numpy()
                        .squeeze(0)
                    )
                    # Concatenate one_hot_label and image_embedding
                    self.List_inputs.append(np.concatenate((one_hot_label, image_embedding)))

            elif self.import_criterions == "CBM_SIGLIP_weighted_sum+SIGLIP_heatmap+labels":
                activations = activations_pth
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = tokenizer(text).to(self.device)
                    text_embedding = (
                        self.openclip_model.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    # Concatenate one_hot_label and text_embedding

                    weighted_embeds = np.sum(
                        np.array(
                            [
                                activations[i] * d_clips_embed_concepts[dataset_name][i]
                                for i in range(len(activations))
                            ]
                        ),
                        axis=0,
                    )
                    # Concat one_hot_label and text_embeding
                    self.List_inputs.append(np.concatenate((one_hot_label, weighted_embeds)))
                elif activations.endswith(".npy"):
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(explication)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.openclip_model.encode_image(img_tensor)
                        .cpu()
                        .detach()
                        .numpy()
                        .squeeze(0)
                    )
                    # Concatenate one_hot_label and image_embedding
                    self.List_inputs.append(np.concatenate((one_hot_label, image_embedding)))

            elif self.import_criterions == "SIGLIP_CBM_text+SIGLIP_image_blur+labels":
                activations = activations_pth
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = tokenizer(text).to(self.device)
                    text_embedding = (
                        self.openclip_model.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    # Concatenate one_hot_label and text_embedding
                    self.List_inputs.append(np.concatenate((one_hot_label, text_embedding)))
                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    activations = np.clip(np.load(activations_pth), 0, 1)
                    if xai_id in ["46", "47"]:
                        activations = activations[:, :, -1]
                    image_blur = input_image * activations[:, :, np.newaxis]
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(image_blur)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.openclip_model.encode_image(img_tensor)
                        .cpu()
                        .detach()
                        .numpy()
                        .squeeze(0)
                    )
                    # Concatenate one_hot_label and image_embedding
                    self.List_inputs.append(np.concatenate((one_hot_label, image_embedding)))

            elif self.import_criterions == "CLIP_CBM_text+CLIP_heatmap":
                activations = activations_pth
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = clip.tokenize(text).to(self.device)
                    text_embeding = (
                        self.clip_net.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(text_embeding)
                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(explication)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(image_embedding)

            elif (self.import_criterions == "LLaVa_CBM_text+LLaVa_heatmap") and (
                flag_saved is False
            ):
                activations = activations_pth

                if Path(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy").exists():
                    self.List_inputs.append(
                        np.load(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy")
                    )

                else:
                    if activations.endswith(".json"):
                        with Path(activations).open("r") as fp:
                            activations_dict = json.load(fp)
                        dataset_name = entry["img"].split("/")[2].split("_")[0]
                        activations = utils.activations_from_dict(
                            activations_dict, dataset_name, xai_id
                        )
                        text = utils.sentence_from_activations(
                            activations, dataset_name, xai_id, top_n=param_cbm
                        )
                        text_embeding = self.llava_net.get_text_embedding(text)
                        self.List_inputs.append(text_embeding)

                        with Path(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy").open(
                            "wb"
                        ) as f:
                            np.save(f, np.array(text_embeding))

                    elif activations.endswith(".npy"):
                        input_image = np.array(Image.open(self.root+entry["img"]))
                        image_embedding = self.llava_net.get_text_embedding(entry["explication"])
                        self.List_inputs.append(image_embedding)

                        with Path(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy").open(
                            "wb"
                        ) as f:
                            np.save(f, np.array(image_embedding))

            elif (self.import_criterions == "LLaVa_CBM_text+LLaVa_heatmap+labels") and (
                flag_saved is False
            ):
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)
                activations = activations_pth

                if Path(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy").exists():
                    self.List_inputs.append(
                        np.load(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy")
                    )

                else:
                    if activations.endswith(".json"):
                        with Path(activations).open("r") as fp:
                            activations_dict = json.load(fp)
                        dataset_name = entry["img"].split("/")[2].split("_")[0]
                        activations = utils.activations_from_dict(
                            activations_dict, dataset_name, xai_id
                        )
                        text = utils.sentence_from_activations(
                            activations, dataset_name, xai_id, top_n=param_cbm
                        )
                        text_embeding = self.llava_net.get_text_embedding(text)
                        self.List_inputs.append(
                            np.concatenate((one_hot_label, text_embeding.cpu().detach().numpy()))
                        )

                        with Path(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy").open(
                            "wb"
                        ) as f:
                            np.save(f, np.array(text_embeding))

                    elif activations.endswith(".npy"):
                        input_image = np.array(Image.open(self.root+entry["img"]))
                        image_embedding = self.llava_net.get_text_embedding(entry["explication"])
                        self.List_inputs.append(
                            np.concatenate((one_hot_label, image_embedding.cpu().detach().numpy()))
                        )
                        with Path(f"results/results_llava/embed_llava_{img_id}_{xai_id}.npy").open(
                            "wb"
                        ) as f:
                            np.save(f, np.array(image_embedding))

            elif self.import_criterions == "CLIP_CBM_text+CLIP_heatmap+labels":
                activations = activations_pth
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)

                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = clip.tokenize(text).to(self.device)
                    text_embeding = (
                        self.clip_net.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )

                    # Concat one_hot_label and text_embeding
                    self.List_inputs.append(np.concatenate((one_hot_label, text_embeding)))

                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(explication)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                    )
                    # Concat one_hot_label and text_embeding
                    self.List_inputs.append(np.concatenate((one_hot_label, image_embedding)))

            elif self.import_criterions == "CBM_CLIP_weighted_sum+CLIP_heatmap+labels":
                activations = activations_pth
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)
                if activations.endswith(".json"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    with Path(activations_pth).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id, normalize=True
                    )
                    weighted_embeds = np.sum(
                        np.array(
                            [
                                activations[i] * d_clips_embed_concepts[dataset_name][i]
                                for i in range(len(activations))
                            ]
                        ),
                        axis=0,
                    )
                    # Concat one_hot_label and text_embeding
                    self.List_inputs.append(np.concatenate((one_hot_label, weighted_embeds)))
                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    explication = np.array(Image.open(self.root+entry["explication"]))
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(explication)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(np.concatenate((one_hot_label, image_embedding)))

            elif self.import_criterions == "CLIP_CBM_text+CLIP_image_blur":
                activations = activations_pth
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = clip.tokenize(text).to(self.device)
                    text_embeding = (
                        self.clip_net.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(text_embeding)
                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    activations = np.clip(np.load(activations_pth), 0, 1)
                    if xai_id in ["46", "47"]:
                        activations = activations[:, :, -1]
                    image_blur = input_image * activations[:, :, np.newaxis]
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(image_blur)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(image_embedding)

            elif self.import_criterions == "CLIP_CBM_text+CLIP_image_blur+labels":
                activations = activations_pth
                label = entry["img"].split("/")[-1].split("_")[-1].split(".")[0]
                one_hot_label = utils.convert_label_to_one_hot(label)
                if activations.endswith(".json"):
                    with Path(activations).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id
                    )
                    text = utils.sentence_from_activations(
                        activations, dataset_name, xai_id, top_n=param_cbm
                    )
                    tokens = clip.tokenize(text).to(self.device)
                    text_embeding = (
                        self.clip_net.encode_text(tokens).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(np.concatenate((one_hot_label, text_embeding)))
                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    activations = np.clip(np.load(activations_pth), 0, 1)
                    if xai_id in ["46", "47"]:
                        activations = activations[:, :, -1]
                    image_blur = input_image * activations[:, :, np.newaxis]
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(image_blur)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(np.concatenate((one_hot_label, image_embedding)))

            elif self.import_criterions == "CBM_CLIP_weighted_sum+CLIP_image_blur":
                activations = activations_pth
                if activations.endswith(".json"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    with Path(activations_pth).open("r") as fp:
                        activations_dict = json.load(fp)
                    dataset_name = entry["img"].split("/")[2].split("_")[0]
                    activations = utils.activations_from_dict(
                        activations_dict, dataset_name, xai_id, normalize=True
                    )
                    weighted_embeds = np.sum(
                        np.array(
                            [
                                activations[i] * d_clips_embed_concepts[dataset_name][i]
                                for i in range(len(activations))
                            ]
                        ),
                        axis=0,
                    )
                    self.List_inputs.append(weighted_embeds)
                elif activations.endswith(".npy"):
                    input_image = np.array(Image.open(self.root+entry["img"]))
                    activations = np.clip(np.load(activations_pth), 0, 1)
                    if xai_id in ["46", "47"]:
                        activations = activations[:, :, -1]
                    image_blur = input_image * activations[:, :, np.newaxis]
                    img_tensor = (
                        self.preprocess(Image.fromarray(np.uint8(image_blur)))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    image_embedding = (
                        self.clip_net.encode_image(img_tensor).cpu().detach().numpy().squeeze(0)
                    )
                    self.List_inputs.append(image_embedding)

            elif self.import_criterions == "metrics":
                self.List_inputs.append(
                    [
                        float(self.dict_criterions[f"{img_id}_{xai_id}"][key])
                        for key in select_criterions
                    ]
                )

        if self.import_criterions:
            self.List_inputs = np.array(self.List_inputs)
            self.List_outputs = (
                np.array(self.List_label) - 1
            ) / 4  # Convert notes to 1 to 5 to 0 to 1

        if callibrate_mode:
            self.calibrate_labels()

    def calibrate_labels(self):
        # Dictionary to accumulate labels and their positions for each img_id
        img_id_groups = defaultdict(list)

        desired_average = np.mean(self.List_label)

        # Group labels by img_id
        for idx, (img_id, label) in enumerate(zip(self.List_img_id, self.List_label)):
            img_id_groups[img_id].append((idx, label))

        # Calculate translations and apply them
        for group in img_id_groups.values():
            indices, labels = zip(*group)
            current_avg = sum(labels) / len(labels)
            translation = desired_average - current_avg

            # Apply translation to each label in the group
            for idx in indices:
                self.List_label[idx] += translation

    def __getitem__(self, index):
        """Returns a dictionary containing the image path, the input image array, the id of the sample, and its label.

        Args:
            index (int): The index of the sample in the dataset.

        Returns:
            dict: A dictionary with the following keys: "path" (str), "image" (ndarray), "id_sample" (str), "label" (str).
        """
        if self.List_xai_id[index] in ["39", "40", "41", "42", "43", "44", "45", "48", "49"]:
            with Path(self.List_act_pth[index]).open("r") as fp:
                activations_dict = json.load(fp)
            dataset_name = self.List_img_pth[index].split("/")[2].split("_")[0]
            activations = utils.activations_from_dict(
                activations_dict, dataset_name, self.List_xai_id[index]
            )
        else:
            with Path(self.List_act_pth[index]).open("rb") as fp:
                activations = np.load(fp)
            if self.List_xai_id[index] in ["46", "47"]:
                activations = activations[:, :, -1]

        label = np.array(self.List_label[index])
        input_image = np.array(Image.open(self.List_img_pth[index]))

        if self.import_criterions:
            a_criterions = np.array(
                self.dict_criterions[
                    f"{self.List_img_id[index]}_{self.List_xai_id[index]}".values()
                ]
            )

            # Since not present in the 2nd batch, i removed
            """"bboxes": self.List_bboxes[index],
            "description": self.List_descriptions[index],"""

            return {
                "image": input_image,
                "activations": activations,
                "label": label,
                "xai_id": self.List_xai_id[index],
                "img_id": self.List_img_id[index],
                "criterions": a_criterions,
                "GT_class": self.List_gtclass[index],
                "act_pth": self.List_act_pth[index],
            }

        return {
            "image": input_image,
            "activations": activations,
            "label": label,
            "xai_id": self.List_xai_id[index],
            "img_id": self.List_img_id[index],
            "GT_class": self.List_gtclass[index],
            "act_pth": self.List_act_pth[index],
        }

    def find_matching_entries(self, data, idimg_idxai_set):
        for entry in data:
            explication_id = self.extract_ids_from_explication(entry["explication"])
            if explication_id == idimg_idxai_set:
                return (
                    entry["img"],
                    entry["explication"],
                    entry["explication"].replace(".png", ".npy").replace("/expl/", "/activations/"),
                    entry["GTclass"],
                    entry["predictedclass"],
                )

        ValueError("No matching entry found")
        return None

    def extract_ids_from_explication(self, field_value):
        parts = field_value.split("_")
        if len(parts) > 4:  # Ensure there are enough parts to match the required pattern
            idimg = parts[2]
            idxai = parts[3]
            return f"{idimg}_{idxai}"
        return None

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.List_img_pth)

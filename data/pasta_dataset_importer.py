import torch

from data.pasta_dataset import PASTAdataset

def pasta_dataloader_importer(
    dataset_name,
    seed=493,
    type_task="classification",
    import_all=False,
    sub_dataset="catsdogscars",
    import_criterions="",
    device="",
    type_expl="saliency",
    sigma_noise="",
    question_id="Q1",
    label_type="mean",
    restrict_split=False,
    position_oracle=0,
    param_cbm=False,
    file_annotations="human_annotations.json",
):
    """Import a sample dataloader based on the dataset name.

    Args:
        dataset_name (str): The name of the dataset.
        seed (int, optional): The seed of the dataset shuffle. Defaults to 493.
        type_task (str, optional): The type of the task. Defaults to 'classification'.
        import_all (bool, optional): Whether to import all the samples. Defaults to False.
        sub_dataset (str, optional): The sub dataset. Defaults to 'catsdogscars'.
        import_criterions (str, optional): The nature of the input criterions to put to train PASTA, in [metrics,CLIP_image_blur,CLIP_heatmap,CBM_activations].
        device (str, optional): The device to use. Defaults to "".
        type_expl (str, optional): The type of explanation to deal with in [saliency,cbm]. Defaults to "saliency".
        sigma_noise (str, optional): The sigma of the noise in pasta_toy. Defaults to "".
        question_id (str, optional): The question id. Defaults to "Q1".
        label_type (str, optional): The type of label. Defaults to "mean".
        restrict_split (bool, optional): Whether to restrict the split. Defaults to False.
        position_oracle (int, optional): The position oracle. Defaults to 0.
        param_cbm (bool, optional): Whether to use the param cbm. Defaults to False.
        file_annotations (str, optional): The file containing the human annotations. Defaults to "human_annotations.json".

    Returns:
        torch.utils.data.DataLoader: The dataloader for the sample dataset.
    """

    if dataset_name == "true":
        if import_all:
            dataset = PASTAdataset(
                phase="all",
                import_criterions=import_criterions,
                seed=seed,
                dataset=sub_dataset,
                type_expl="",
            )
            return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)

        dataset_train = PASTAdataset(
            phase="train",
            import_criterions=import_criterions,
            seed=seed,
            dataset=sub_dataset,
            type_expl="",
        )
        dataset_test = PASTAdataset(
            phase="test",
            import_criterions=import_criterions,
            seed=seed,
            dataset=sub_dataset,
            type_expl="",
        )
        return (
            torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=1),
            None,
            torch.utils.data.DataLoader(dataset_test, shuffle=True, batch_size=1),
        )

    if dataset_name == "true_sklearn":

        dataset_train = PASTAdataset(
            phase="train",
            import_criterions=import_criterions,
            seed=seed,
            type_task=type_task,
            dataset=sub_dataset,
            device=device,
            type_expl=type_expl,
            question_id=question_id,
            label_type=label_type,
            restrict_split=restrict_split,
            position_oracle=position_oracle,
            param_cbm=param_cbm,
            file_annotations=file_annotations,
        )

        dataset_val = PASTAdataset(
            phase="val",
            import_criterions=import_criterions,
            seed=seed,
            type_task=type_task,
            dataset=sub_dataset,
            device=device,
            type_expl=type_expl,
            question_id=question_id,
            label_type=label_type,
            restrict_split=restrict_split,
            position_oracle=position_oracle,
            param_cbm=param_cbm,
            file_annotations=file_annotations,
        )

        dataset_test = PASTAdataset(
            phase="test",
            import_criterions=import_criterions,
            seed=seed,
            type_task=type_task,
            dataset=sub_dataset,
            device=device,
            type_expl=type_expl,
            question_id=question_id,
            label_type=label_type,
            restrict_split=restrict_split,
            position_oracle=position_oracle,
            param_cbm=param_cbm,
            file_annotations=file_annotations,
        )

        return dataset_train, dataset_val, dataset_test

    raise NotImplementedError

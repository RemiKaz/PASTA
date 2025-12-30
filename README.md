# PASTA Instructions

This repository contains code for two main components of the paper: Benchmarking XAI Explanations with Human-Aligned Evaluations. For more details, see the paper: [Benchmarking XAI Explanations with Human-Aligned Evaluations](https://arxiv.org/abs/2411.02470v2).

The PASTA dataset is available on Hugging Face: [PASTA Dataset](https://huggingface.co/datasets/ENSTA-U2IS/PASTA/tree/main).

## Explanation Computation Pipeline

To generate explanations using the XAI methods included in the PASTA dataset:

1. Download the classifier's weights from: [PASTA Weights on Hugging Face](https://huggingface.co/ENSTA-U2IS/PASTA_weights)
2. Place the downloaded weights in the `models_pkl` folder
3. Run the explanation computation script:

```bash
python compute_explanation.py --network CLASSIFIER_NAME --expl_method XAI_METHOD --input_dataset CLASSIFICATION_DATASET --input_image_path PATH_OF_THE_IMG_TO_EXPLAIN
```

Example:
```bash
python compute_explanation.py --network resnet50 --expl_method GradCAM --input_dataset monumai --input_image_path data_samples/Monumai/Images/MonuMAI_6_Baroque.png
```

## Implemented Methods

## PASTA Score Calculation

To compute the PASTA score for a new XAI method:

1. Download the test set images
2. Generate explanations for all images in the test set (test_imgs_PASTA_score.zip in the [PASTA Dataset link](https://huggingface.co/datasets/ENSTA-U2IS/PASTA/tree/main)):
   - For saliency-based explanations, save the resulting heatmap images
   - For CBM-based explanations, save JSON dictionaries containing concept values
   - File format should be: `{ROOT}/{DATASET}_{ID_IMG}_{LABEL}.{EXTENTION}`
3. Run the PASTA score script:

```bash
python pasta_score.py --path_explanation PATH_TO_THE_EXPLANATIONS --id_question ID_PASTA_QUESTION --out_format TYPE_OF_EXPLANATION
```
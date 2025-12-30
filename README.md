# Project page PASTA (Perceptual Assessment System for explanaTion of Artificial Intelligence)

This repository contains code for two main components of the paper: Benchmarking XAI Explanations with Human-Aligned Evaluations. For more details, see the paper: [Benchmarking XAI Explanations with Human-Aligned Evaluations](https://arxiv.org/abs/2411.02470v2).

The PASTA dataset is available on Hugging Face: [PASTA Dataset](https://huggingface.co/datasets/ENSTA-U2IS/PASTA/tree/main).

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

### Implemented Methods

The following table summarizes the XAI methods included in the PASTA dataset:

| Name                          | Functioning               | Attribution on | Stage    | Applied on                          |
|-------------------------------|---------------------------|----------------|----------|-------------------------------------|
| BCos                          | Interpretable latent space| Image          | Ante-hoc | ResNet50-BCos                       |
| GradCAM                       | Gradient                  | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| HiResCAM                      | Gradient                  | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| GradCAMElementWise            | Gradient                  | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| GradCAM++                     | Gradient                  | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| XGradCAM                      | Gradient                  | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| AblationCAM                   | Perturbation               | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| ScoreCAM                      | Perturbation               | Image          | Post-hoc | ViT, ResNet50                       |
| EigenCAM                      | Factorization              | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| EigenGradCAM                  | Gradient+Factorization     | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| FullGrad                      | Gradient                  | Image          | Post-hoc | ViT, ResNet50                       |
| Deep Feature Factorizations    | Factorization              | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| SHAP                          | Perturbation               | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| LIME                          | Perturbation               | Image          | Post-hoc | ViT, ResNet50, CLIP (zero-shot)     |
| X-NeSyL                       | Interpretable latent space | Concepts       | Ante-hoc | X-NeSyL                             |
| CLIP-linear-sample            | Interpretable latent space | Concepts       | Ante-hoc | CLIP-linear                         |
| CLIP-QDA-sample               | Counterfactual             | Concepts       | Ante-hoc | CLIP-QDA                            |
| LIME-CBM                      | Perturbation               | Concepts       | Post-hoc | CLIP-QDA, ConceptBottleneck         |
| SHAP-CBM                      | Perturbation               | Concepts       | Post-hoc | CLIP-QDA, ConceptBottleneck         |
| RISE-CBM                      | Perturbation               | Concepts       | Post-hoc | ConceptBottleneck                   |


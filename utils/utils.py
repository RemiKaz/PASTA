import json
import random
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

import data


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,  # Whether to use an RGB or BGR heatmap; set to True if 'img' is in RGB format.
    image_weight: float = 0.5,  # The final result is image_weight * img + (1-image_weight) * mask.
) -> np.ndarray:
    """Overlays the cam mask on the image as a heatmap.
    By default, the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap.
    :param image_weight: The weight of the image in the final result.
    :returns: The default image with the cam overlay.
    :raises ValueError: If the input image is not in the range [0, 1] or if image_weight is not in the range [0, 1].
    """
    # Define the colormap and get the heatmap
    cm_bwr = plt.get_cmap("bwr")
    heatmap = cm_bwr(mask)[:, :, 0:-1]
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Check if the input image is in the range [0, 1]
    if np.max(img) > 1:
        raise ValueError("The input image should be np.float32 in the range [0, 1]")

    # Check if image_weight is in the range [0, 1]
    if image_weight < 0 or image_weight > 1:
        raise ValueError(f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    # Compute the final result and normalize it
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def colors_from_values(values, palette_name):
    """Given a list of values and a colormap name, this function returns a list
    of corresponding RGB colors.

    Parameters
    ----------
    values : array-like
        The list of values to be mapped to colors.
    palette_name : str
        The name of the colormap to use.

    Returns:
    -------
    array-like
        The list of colors, each represented as an RGB tuple.
    """
    # Remove duplicate values

    if isinstance(values[0], np.ndarray):
        values = [tuple(arr) for arr in values]

    no_dup_values = list(set(values))

    if len(no_dup_values) == len(values):
        # Normalize the values to range [0, 1]
        abs_max = max(np.abs(values))
        normalized = (values / abs_max + 1) * 0.5

        # Convert to indices
        indices = np.round(normalized * (len(values) - 1)).astype(np.int32)

        # Use the indices to get the colors
        palette = sns.color_palette(palette_name, len(values))

        # Take the colors from the palette using the indices

        if len(np.array(palette).shape) == 3:
            return list(np.array(palette).take(indices, axis=0)[:, 0, :])

        return list(np.array(palette).take(indices, axis=0))

    # Normalize the values to range [0, 1]
    abs_max = max(np.abs(no_dup_values))
    normalized = (no_dup_values / abs_max + 1) * 0.5

    # Convert to indices
    indices = np.round(normalized * (len(no_dup_values))).astype(np.int32)

    # Use the indices to get the colors
    palette = sns.color_palette(palette_name, len(no_dup_values))

    dict_palette = dict(zip(sorted(no_dup_values), palette))

    return [dict_palette[value] for value in values]


def randomize_rectangle(img_size, max_length, max_width):
    """Generate random coordinates for a rectangle within an image.

    Args:
        img_size (tuple): The size of the image as (width, height).
        max_length (int): The maximum length of the rectangle.
        max_width (int): The maximum width of the rectangle.

    Returns:
        tuple: The coordinates of the rectangle (x1, y1, x2, y2).
    """
    # Generate random x1 and y1 coordinates
    x1 = random.randint(0, img_size[0])
    y1 = random.randint(0, img_size[1])

    # Compute max_x2 and max_y2 coordinates
    max_x2 = min(img_size[0], x1 + max_length)
    max_y2 = min(img_size[1], y1 + max_width)

    # Generate random x2 and y2 coordinates within the rectangle
    x2 = random.randint(x1, max_x2)
    y2 = random.randint(y1, max_y2)

    return (x1, y1, x2, y2)


def mask_image(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply a mask to an image and mask the part outside the mask with a black box.

    Args:
            img (PIL.Image.Image): The original image.
            mask (PIL.Image.Image): The mask image. It should have the same dimensions as img.

    Returns:
            PIL.Image.Image: The resulting image with the masked part and black box outside the mask.
    """
    # Convert images to RGBA mode
    img_rgba = img.convert("RGBA")
    mask_rgba = mask

    # Create a black box image with the same size as the original image
    black_box = Image.new("RGBA", img_rgba.size, color=(0, 0, 0, 255))

    # Composite the original image and the black box using the mask
    result_img = Image.composite(black_box, img_rgba, mask_rgba)
    # Convert the result back to RGB mode
    return result_img.convert("RGB")


def reshape_for_tensor(img: np.ndarray) -> np.ndarray:
    """Reshape an image to be compatible with a tensor.

    Args:
            img (np.ndarray): The image to be reshaped.
    """
    if img.shape[-1] == 3:
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        elif len(img.shape) == 4:
            img = img.transpose(0, 3, 1, 2)

    return img


def delete_id(xai_id_to_delete, dataset, data_json):
    filtered_data = []
    for entry in data_json:
        id_xai = entry["explication"].split("_")[3]
        dataset_name = entry["explication"].split("_")[1].split("/")[-1]
        if (str(id_xai) == str(xai_id_to_delete)) and (dataset_name == dataset):
            # Delete related files
            if Path(entry["explication"]).exists():
                Path(entry["explication"]).unlink()
            if Path(entry["activations"]).exists():
                Path(entry["activations"]).unlink()
            if Path(entry["explicationpertubated"]).exists():
                Path(entry["explicationpertubated"]).unlink()
            if Path(entry["activationspertubated"]).exists():
                Path(entry["activationspertubated"]).unlink()
            if Path(entry["imgpertubated"]).exists():
                Path(entry["imgpertubated"]).unlink()
        else:
            filtered_data.append(entry)

    print(f"Deleted {len(data_json) - len(filtered_data)} entries")

    return filtered_data


def save_as_npy(list_to_save, path):
    """Save a list as a .npy file.

    Args:
        list_to_save (list): The list to save.
        path (str): The path of the .npy file.
    """
    with Path(path).open("wb") as f:
        np.save(f, list_to_save)


def save_as_json(dict_to_save, path):
    """Save a dictionary as a .json file.

    Args:
        dict_to_save (dict): The dictionary to save.
        path (str): The path of the .json file.
    """
    with Path(path).open("w") as outfile:
        json.dump(dict_to_save, outfile)


def extract_str(test_str, sub1, sub2):
    """Extract a substring from a string between two substrings.

    Args:
        test_str (str): The string to extract the substring from.
        sub1 (str): The substring to start the extraction.
        sub2 (str): The substring to end the extraction.
    """
    # getting index of substrings
    idx1 = test_str.index(sub1)
    idx2 = test_str.index(sub2)

    # length of substring 1 is added to
    # get string from next character
    return test_str[idx1 + len(sub1) + 1 : idx2]

    # printing result


def plot_image_with_bboxes(image, adjusted_bboxes, name_file):
    # Load and plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot each bounding box
    for bbox in adjusted_bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    plt.savefig(name_file)


def xai_id_from_model_expl(expl_name, model_name):
    # Define explanation
    if (
        model_name == "CLIP-zero-shot"
        or model_name == "resnet50"
        or model_name == "vitB"
        or (model_name == "CLIP-QDA" and expl_name != "CLIP-QDA-sample")
        or model_name == "CBM-classifier-logistic"
        or model_name == "resnet50-bcos"
        or model_name == "vitB-bcos"
    ):
        expl_method_full = expl_name + f" ({model_name})"

    else:
        expl_method_full = expl_name

    # Methods available
    list_xai_methods = [
        "GradCAM (resnet50)",
        "GradCAM (vitB)",
        "GradCAM (CLIP-zero-shot)",
        "LIME (resnet50)",
        "LIME (vitB)",
        "LIME (CLIP-zero-shot)",
        "SHAP (resnet50)",
        "SHAP (vitB)",
        "SHAP (CLIP-zero-shot)",
        "AblationCAM (resnet50)",
        "AblationCAM (vitB)",
        "AblationCAM (CLIP-zero-shot)",
        "EigenCAM (resnet50)",
        "EigenCAM (vitB)",
        "EigenCAM (CLIP-zero-shot)",
        "EigenGradCAM (resnet50)",
        "EigenGradCAM (vitB)",
        "EigenGradCAM (CLIP-zero-shot)",
        "FullGrad (resnet50)",
        "FullGrad (vitB)",
        "FullGrad (CLIP-zero-shot)",
        "GradCAMPlusPlus (resnet50)",
        "GradCAMPlusPlus (vitB)",
        "GradCAMPlusPlus (CLIP-zero-shot)",
        "GradCAMElementWise (resnet50)",
        "GradCAMElementWise (vitB)",
        "GradCAMElementWise (CLIP-zero-shot)",
        "HiResCAM (resnet50)",
        "HiResCAM (vitB)",
        "HiResCAM (CLIP-zero-shot)",
        "ScoreCAM (resnet50)",
        "ScoreCAM (vitB)",
        "ScoreCAM (CLIP-zero-shot)",
        "XGradCAM (resnet50)",
        "XGradCAM (vitB)",
        "XGradCAM (CLIP-zero-shot)",
        "DeepFeatureFactorization (resnet50)",
        "DeepFeatureFactorization (vitB)",
        "DeepFeatureFactorization (CLIP-zero-shot)",
        "CLIP-QDA-sample",
        "CLIP-Linear-sample",
        "LIME_CBM (CLIP-QDA)",
        "SHAP_CBM (CLIP-QDA)",
        "LIME_CBM (CBM-classifier-logistic)",
        "SHAP_CBM (CBM-classifier-logistic)",
        "Xnesyl-Linear",
        "BCos (resnet50-bcos)",
        "BCos (vitB-bcos)",
        "Rise_CBM (CLIP-QDA)",
        "Rise_CBM (CBM-classifier-logistic)",
    ]

    return list_xai_methods.index(expl_method_full)


def activations_from_dict(dict_activations, dataset, xai_id, normalize=False):
    list_concepts = data.metadata_importer(dataset)["labeled_concepts"]

    if xai_id == "39":  # CLIP-QDA-sample
        array_activations = np.ones(len(list_concepts)) * 10000

    else:
        array_activations = np.zeros(len(list_concepts))

    if xai_id == "39":  # CLIP-QDA-sample
        for i_concept, concept in enumerate(list_concepts):
            for key in dict_activations:
                if (
                    concept in key.split("(")[0]
                    and np.abs(dict_activations[key]) > array_activations[i_concept]
                ):
                    array_activations[i_concept] = np.array(dict_activations[key])

    else:
        for i_concept, concept in enumerate(list_concepts):
            for key in dict_activations:
                if concept in key:
                    array_activations[i_concept] = np.array(dict_activations[key])

    if normalize:
        if np.sum(np.abs(array_activations)) == 0:
            return array_activations
        array_activations_abs = np.abs(array_activations)
        if xai_id == "39":
            array_activations_abs = 1 / (array_activations_abs + 0.001)
        array_activations = array_activations_abs / np.linalg.norm(array_activations_abs)

    return array_activations


def sentence_from_activations(activations, dataset, xai_id, top_n=False, keep_sign="all"):
    list_concepts = data.metadata_importer(dataset)["labeled_concepts"]

    '''sentence = ""'''
    sentence = ""

    if keep_sign == "positive":
        # Discard negative activations
        activations = np.where(activations < 0, 0, activations)

    elif keep_sign == "negative":
        # Discard positive activations
        activations = np.where(activations > 0, 0, activations)

    # Top N concepts

    if xai_id == "39":
        _, concept_sorted = zip(*sorted(zip(np.abs(activations), list_concepts), reverse=False))

    else:
        _, concept_sorted = zip(*sorted(zip(np.abs(activations), list_concepts), reverse=True))

    top_concepts = concept_sorted[:top_n] if top_n else concept_sorted

    for concept in top_concepts:
        sentence += concept + ", "

    return sentence


def modify_activations_blur(activations, mode="treshold", treshold_value=0.5):
    if mode == "treshold":
        binary_mask = np.where(activations >= treshold_value, 1, 0)
    if mode == "sigmoid":
        activations = 1 / (1 + np.exp(-activations))
    return binary_mask

def convert_label_to_one_hot(label_text):
    list_labels = [
        "shopping_and_dining",
        "workplace",
        "home_or_hotel",
        "transportation",
        "sports_and_leisure",
        "cultural",
        "Baroque",
        "Gothic",
        "Hispanic-Muslim",
        "Renaissance",
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "cat",
        "cow",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    ]

    if label_text == "cats":
        label_text = "cat"
    elif label_text == "dogs":
        label_text = "dog"
    elif label_text == "cars":
        label_text = "car"
    elif label_text == "hotel":
        label_text = "home_or_hotel"
    elif label_text == "dining":
        label_text = "shopping_and_dining"
    elif label_text == "leisure":
        label_text = "sports_and_leisure"

    id_label = list_labels.index(label_text)

    # Convert to one hot vector

    one_hot_vector = np.zeros(len(list_labels))
    one_hot_vector[id_label] = 1

    return one_hot_vector


def convert_dataset_to_one_hot(label_text):
    list_datasets = ["coco", "pascalpart", "monumai", "catsdogscars"]

    id_label = list_datasets.index(label_text)

    # Convert to one hot vector

    one_hot_vector = np.zeros(len(list_datasets))
    one_hot_vector[id_label] = 1

    return one_hot_vector

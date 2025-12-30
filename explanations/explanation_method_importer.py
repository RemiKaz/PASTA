from explanations.ablationcam import AblationCAM
from explanations.bcos import BCos
from explanations.clip_labo_sample import CLIPLaBosample
from explanations.clip_linear_sample import CLIPLinearSample
from explanations.clip_qda_sample import CLIPQDAsample
from explanations.deepfeaturefactorization import DeepFeatureFactorization
from explanations.eigencam import EigenCAM
from explanations.eigengradcam import EigenGradCAM
from explanations.fullgrad import FullGrad
from explanations.gradcam import GradCAM
from explanations.gradcamelementwise import GradCAMElementWise
from explanations.gradcamplusplus import GradCAMPlusPlus
from explanations.guided_backprop import GuidedBackpropReLUModel
from explanations.hirescam import HiResCAM
from explanations.lime_cbm import LIMECBM
from explanations.lime_image import LIMEimage
from explanations.rise import Rise
from explanations.rise_image import RiseImage
from explanations.scorecam import ScoreCAM
from explanations.shap_cbm import SHAPCBM
from explanations.shap_image import SHAPimage
from explanations.xgradcam import XGradCAM
from explanations.xnesyl_linear_sample import XnesylLinearSample


def explanation_method_importer(explanation_name, **kwargs):
    """Import and return the chosen explanation method based on the model name and chosen explanation name.

    Args:
        explanation_name (str): Name of the chosen explanation method.
        kwargs (dict): Dictionary containing the model name and model/device.

    Returns:
        explanation_method: The chosen explanation method.

    Raises:
        ValueError: If the chosen model name or explanation name is not implemented.
    """
    model_name = kwargs["model_name"]

    print("explanation_name", explanation_name)

    if (model_name == "resnet50") or (model_name == "vitB") or (model_name == "CLIP-zero-shot"):
        model, device = kwargs["model"], kwargs["device"]
        if explanation_name == "LIME":
            """explanation_method = LIMEimage(kwargs["dict_hyperparam"])"""
            explanation_method = LIMEimage()
        elif explanation_name == "SHAP":
            explanation_method = SHAPimage(model, device)
        elif explanation_name == "GradCAM":
            """explanation_method = GradCAM(model, device, kwargs["dict_hyperparam"])"""
            explanation_method = GradCAM(model, device)
            print("No dict_hyperparam for GradCAM")
        elif explanation_name == "AblationCAM":
            explanation_method = AblationCAM(model, device)
        elif explanation_name == "EigenCAM":
            explanation_method = EigenCAM(model, device)
        elif explanation_name == "EigenGradCAM":
            explanation_method = EigenGradCAM(model, device)
        elif explanation_name == "FullGrad":
            explanation_method = FullGrad(model, device)
        elif explanation_name == "GradCAMPlusPlus":
            explanation_method = GradCAMPlusPlus(model, device)
        elif explanation_name == "GradCAMElementWise":
            """explanation_method = GradCAMElementWise(model, device, kwargs["dict_hyperparam"])"""
            print("No dict_hyperparam for GradCAMElementWise")
            explanation_method = GradCAMElementWise(model, device)
        elif explanation_name == "HiResCAM":
            explanation_method = HiResCAM(model, device)
        elif explanation_name == "ScoreCAM":
            explanation_method = ScoreCAM(model, device)
        elif explanation_name == "XGradCAM":
            explanation_method = XGradCAM(model, device)
        elif explanation_name == "DeepFeatureFactorization":
            explanation_method = DeepFeatureFactorization(model, device)
        elif explanation_name == "GuidedBackpropReLUModel":
            explanation_method = GuidedBackpropReLUModel(model, device)
        elif explanation_name == "Rise_image":
            explanation_method = RiseImage(model, device)
        elif explanation_name == "Rise_PASTA":
            explanation_method = RiseImage(
                model, device, pasta_mode=True, lambda_pasta=kwargs["lambda_pasta"]
            )
        else:
            raise ValueError("Method not implemented !")

    elif model_name == "resnet50-bcos" or model_name == "vitB-bcos":
        model, device = kwargs["model"], kwargs["device"]
        if explanation_name == "BCos":
            explanation_method = BCos(model, device)
        else:
            raise ValueError("Method not implemented !")

    elif model_name == "CLIP-QDA":
        model = kwargs["model"]
        if explanation_name == "CLIP-QDA-sample":
            device = kwargs["device"]
            explanation_method = CLIPQDAsample(model, device)
        elif explanation_name == "LIME_CBM":
            explanation_method = LIMECBM(model)
        elif explanation_name == "SHAP_CBM":
            explanation_method = SHAPCBM(model)
        elif explanation_name == "Rise_CBM":
            explanation_method = Rise(model)
        else:
            raise ValueError("Method not implemented !")

    elif model_name == "CBM-classifier-logistic":
        if explanation_name == "LIME_CBM":
            explanation_method = LIMECBM(kwargs["model"])
        elif explanation_name == "SHAP_CBM":
            explanation_method = SHAPCBM(kwargs["model"])
        elif explanation_name == "Rise_CBM":
            explanation_method = Rise(kwargs["model"])
        else:
            raise ValueError("Method not implemented !")

    elif model_name == "CLIP-Linear":
        model = kwargs["model"]
        if explanation_name == "CLIP-Linear-sample":
            explanation_method = CLIPLinearSample(model)

    elif model_name == "XNES-classifier-logistic":
        model, device = kwargs["model"], kwargs["device"]
        if explanation_name == "Xnesyl-Linear":
            explanation_method = XnesylLinearSample(model, device)

    elif model_name == "CLIP-LaBo":
        model = kwargs["model"]
        if explanation_name == "CLIP-LaBo-sample":
            explanation_method = CLIPLaBosample(model)

    else:
        raise ValueError("Method not implemented !")

    return explanation_method

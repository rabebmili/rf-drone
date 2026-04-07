"""Registre centralisé des modèles pour la forensique RF de drones."""

import importlib


MODEL_REGISTRY = {
    "smallrf": ("src.models.cnn_spectrogram", "SmallRFNet"),
    "resnet": ("src.models.resnet_spectrogram", "RFResNet"),
    "transformer": ("src.models.transformer_spectrogram", "RFTransformer"),
    "efficientnet": ("src.models.efficientnet_spectrogram", "RFEfficientNet"),
    "ast": ("src.models.ast_spectrogram", "RFAST"),
    "conformer": ("src.models.conformer_spectrogram", "RFConformer"),
    "cnn1d": ("src.models.cnn_1d", "RFCNN1D"),
}

# Modèles nécessitant un signal 1D brut au lieu de spectrogrammes 2D
RAW_SIGNAL_MODELS = {"cnn1d"}


def get_model(name, num_classes, **kwargs):
    # Importe et instancie un modèle par son nom dans le registre
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    module_path, class_name = MODEL_REGISTRY[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(num_classes=num_classes, **kwargs)

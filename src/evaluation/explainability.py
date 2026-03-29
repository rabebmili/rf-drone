"""Grad-CAM heatmaps on spectrograms for model explainability."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """Grad-CAM for CNN models on spectrograms."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Enregistrer les hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for a single input. Returns (heatmap, class, confidence)."""
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Rétropropagation pour la classe cible
        self.model.zero_grad()
        output[0, target_class].backward()

        # Calculer Grad-CAM
        gradients = self.gradients[0]  # [C, h, w]
        activations = self.activations[0]  # [C, h, w]

        # Pooling moyen global des gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Combinaison pondérée des cartes d'activation
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU et normalisation
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Redimensionner à la taille de l'entrée
        cam_np = cam.cpu().numpy()
        from scipy.ndimage import zoom
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        zoom_h = input_h / cam_np.shape[0]
        zoom_w = input_w / cam_np.shape[1]
        heatmap = zoom(cam_np, (zoom_h, zoom_w), order=1)

        return heatmap, target_class, confidence


def get_target_layer(model, model_name="smallrf"):
    """Get the appropriate target layer for Grad-CAM by model type."""
    if model_name == "smallrf":
        # Derniere couche conv dans le bloc features sequentiel
        return model.features[8]  # 3rd Conv2d
    elif model_name == "resnet":
        # Derniere conv du dernier bloc residuel
        return model.layer3[-1].conv2
    else:
        raise ValueError(f"Grad-CAM target layer not defined for model: {model_name}")


def plot_gradcam(spectrogram, heatmap, predicted_class, confidence,
                 class_names=None, output_path=None, title=None):
    """Plot spectrogram with Grad-CAM overlay."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(10)]

    class_label = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Spectrogramme original
    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Spectrogramme original")
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel("Frequence")

    # Carte de chaleur Grad-CAM
    axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_title("Carte de chaleur Grad-CAM")
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("Frequence")

    # Superposition
    axes[2].imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    axes[2].imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[2].set_title(f"Prediction : {class_label} ({confidence:.2%})")
    axes[2].set_xlabel("Temps")
    axes[2].set_ylabel("Frequence")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def generate_gradcam_examples(model, dataset, device, model_name="smallrf",
                               class_names=None, output_dir="outputs/explainability",
                               n_per_class=3):
    """Generate Grad-CAM visualizations for sample spectrograms from each class."""
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collecter les indices par classe
    class_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        c = label.item()
        if c not in class_indices:
            class_indices[c] = []
        if len(class_indices[c]) < n_per_class:
            class_indices[c].append(i)

    for cls, indices in class_indices.items():
        for j, idx in enumerate(indices):
            x, y = dataset[idx]
            x_input = x.unsqueeze(0).to(device)

            heatmap, pred_cls, conf = gradcam.generate(x_input)
            spectrogram = x.squeeze(0).numpy()

            cls_name = class_names[cls] if class_names else f"class_{cls}"
            fname = f"gradcam_{cls_name.replace(' ', '_')}_sample{j}.png"
            plot_gradcam(spectrogram, heatmap, pred_cls, conf,
                         class_names=class_names,
                         output_path=str(out_dir / fname))

"""Cartes de chaleur Grad-CAM sur spectrogrammes pour l'explicabilité des modèles."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """Grad-CAM pour modèles CNN sur spectrogrammes."""

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
        # Générer la carte Grad-CAM pour une entrée. Retourne (heatmap, classe, confiance)
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


class GradCAM1D:
    """Grad-CAM pour modèles CNN-1D sur signaux bruts."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        # Générer la carte Grad-CAM 1D. Retourne (heatmap_1d, classe_pred, confiance)
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients[0]   # [C, L]
        activations = self.activations[0]  # [C, L]

        weights = gradients.mean(dim=1)  # [C]
        cam = torch.zeros(activations.shape[1], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Interpoler à la longueur de l'entrée
        cam_np = cam.cpu().numpy()
        input_len = input_tensor.shape[2]
        from scipy.ndimage import zoom
        heatmap = zoom(cam_np, input_len / len(cam_np), order=1)

        return heatmap, target_class, confidence


class AttentionRollout:
    """Attention rollout pour modèles Transformer (AST, RFTransformer)."""

    def __init__(self, model, model_name="ast"):
        self.model = model
        self.model_name = model_name
        self.attention_maps = []
        self._register_hooks()

    def _register_hooks(self):
        # Accrocher les couches MultiheadAttention pour capturer les poids d'attention
        self.attention_maps = []

        if self.model_name == "ast":
            blocks = self.model.blocks
        elif self.model_name == "transformer":
            blocks = self.model.transformer
        else:
            return

        for block in blocks:
            # Surcharger le forward d'attention pour capturer les poids
            orig_forward = block.attn.forward

            def make_hook(orig_fn):
                def hooked_forward(query, key, value, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = True
                    out, weights = orig_fn(query, key, value, **kwargs)
                    self.attention_maps.append(weights.detach())
                    return out, weights
                return hooked_forward

            block.attn.forward = make_hook(orig_forward)

    def generate(self, input_tensor, target_class=None):
        # Générer la carte d'attention rollout. Retourne (carte_2d, classe_pred, confiance)
        self.attention_maps = []
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)

            if target_class is None:
                target_class = output.argmax(dim=1).item()
            confidence = probs[0, target_class].item()

        if not self.attention_maps:
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((H, W)) * 0.5, target_class, confidence

        # Rollout : multiplier les matrices d'attention à travers les couches
        result = self.attention_maps[0][0].cpu().numpy()  # [T, T]
        for attn in self.attention_maps[1:]:
            attn_np = attn[0].cpu().numpy()
            result = attn_np @ result

        # Extraire l'attention du token CLS vers les patchs
        cls_attention = result[0, 1:]  # attention from CLS to all patches

        # Reformer en grille 2D
        if self.model_name == "ast":
            patch_size = self.model.patch_size
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            grid_h = H // patch_size
            grid_w = W // patch_size
        else:
            grid_h, grid_w = 8, 16  # RFTransformer uses AdaptiveAvgPool2d((8,16))

        n_patches = grid_h * grid_w
        if len(cls_attention) >= n_patches:
            attn_map = cls_attention[:n_patches].reshape(grid_h, grid_w)
        else:
            # Compléter si nécessaire
            padded = np.zeros(n_patches)
            padded[:len(cls_attention)] = cls_attention
            attn_map = padded.reshape(grid_h, grid_w)

        # Normaliser
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Redimensionner aux dimensions de l'entrée
        from scipy.ndimage import zoom
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        zoom_h = H / attn_map.shape[0]
        zoom_w = W / attn_map.shape[1]
        attn_map_resized = zoom(attn_map, (zoom_h, zoom_w), order=1)

        return attn_map_resized, target_class, confidence


def get_target_layer(model, model_name="smallrf"):
    # Obtenir la couche cible appropriée pour Grad-CAM selon le type de modèle
    if model_name == "smallrf":
        # Dernière couche conv dans le séquentiel features
        return model.features[8]  # 3rd Conv2d
    elif model_name == "resnet":
        # Dernière conv du dernier bloc résiduel
        return model.layer3[-1].conv2
    elif model_name == "efficientnet":
        # Dernière Conv2d dans EfficientNet (conv 1x1 avant pooling)
        return model.backbone.features[-1][0]
    elif model_name == "conformer":
        # Dernière Conv2d dans le frontend CNN
        return model.frontend[-3]  # second Conv2d in frontend
    elif model_name == "cnn1d":
        # Dernier bloc Conv1d dans le séquentiel features
        return model.features[-3]  # last Conv1d before final BN+ReLU
    else:
        raise ValueError(f"Grad-CAM target layer not defined for model: {model_name}")


def plot_gradcam(spectrogram, heatmap, predicted_class, confidence,
                 class_names=None, output_path=None, title=None):
    # Tracer le spectrogramme avec superposition Grad-CAM
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
    # Générer les visualisations Grad-CAM ou attention rollout pour des spectrogrammes exemples
    # Choisir la méthode d'explicabilité appropriée
    if model_name in ("ast", "transformer"):
        explainer = AttentionRollout(model, model_name=model_name)
    elif model_name == "cnn1d":
        target_layer = get_target_layer(model, model_name)
        explainer = GradCAM1D(model, target_layer)
    else:
        target_layer = get_target_layer(model, model_name)
        explainer = GradCAM(model, target_layer)

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

            heatmap, pred_cls, conf = explainer.generate(x_input)
            spectrogram = x.squeeze(0).numpy()

            cls_name = class_names[cls] if class_names else f"class_{cls}"
            method = "attention" if model_name in ("ast", "transformer") else "gradcam"
            fname = f"{method}_{cls_name.replace(' ', '_')}_sample{j}.png"
            plot_gradcam(spectrogram, heatmap, pred_cls, conf,
                         class_names=class_names,
                         output_path=str(out_dir / fname))

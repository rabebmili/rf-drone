"""Pipeline forensique intégré combinant classification, open-set, VAE, Siamese, GNN et Grad-CAM."""

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.models import get_model
from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.segmentation import segment_signal
from src.preprocessing.stft_utils import compute_log_spectrogram


class ForensicPipeline:
    """Pipeline d'analyse forensique multi-modèles pour signaux RF de drones."""

    def __init__(self, config):
        # Initialise le pipeline avec la configuration des composants
        self.config = config
        self.device = config.get("device", "cpu")
        self.anomaly_threshold = config.get("anomaly_threshold", 0.7)
        self.class_names = config.get("class_names", ["Background", "Drone"])
        self.num_classes = config.get("num_classes", len(self.class_names))

        # Initialiser les composants
        self.classifier = None
        self.vae = None
        self.siamese = None
        self.openmax_params = None
        self.gallery = None
        self.gnn = None
        self.explainer = None
        self.explainer_type = None

        self._load_classifier()
        self._load_vae()
        self._load_siamese()
        self._load_openmax()
        self._load_gallery()
        self._load_gnn()
        self._load_explainability()

    def _load_classifier(self):
        # Charge le modèle classificateur principal
        weights = self.config.get("classifier_weights")
        if not weights or not Path(weights).exists():
            print("Warning: No classifier weights — classification disabled")
            return

        model_name = self.config.get("classifier_model", "resnet")
        self.classifier = get_model(model_name, num_classes=self.num_classes).to(self.device)
        self.classifier.load_state_dict(
            torch.load(weights, weights_only=True, map_location=self.device)
        )
        self.classifier.eval()
        print(f"Classifier loaded: {model_name} ({weights})")

    def _load_vae(self):
        # Charge le VAE pour la détection d'anomalies
        weights = self.config.get("vae_weights")
        if not weights or not Path(weights).exists():
            return

        from src.models.vae import RFVAE
        latent_dim = self.config.get("vae_latent_dim", 32)
        self.vae = RFVAE(latent_dim=latent_dim).to(self.device)
        self.vae.load_state_dict(
            torch.load(weights, weights_only=True, map_location=self.device)
        )
        self.vae.eval()
        print(f"VAE loaded: latent_dim={latent_dim} ({weights})")

    def _load_siamese(self):
        # Charge le réseau Siamese pour l'attribution par similarité
        weights = self.config.get("siamese_weights")
        if not weights or not Path(weights).exists():
            return

        from src.models.siamese_network import SiameseNetwork
        backbone = self.config.get("siamese_backbone", "resnet")
        self.siamese = SiameseNetwork(
            backbone_name=backbone,
            num_classes=self.num_classes,
        ).to(self.device)
        self.siamese.load_state_dict(
            torch.load(weights, weights_only=True, map_location=self.device)
        )
        self.siamese.eval()
        print(f"Siamese loaded: {backbone} backbone ({weights})")

    def _load_openmax(self):
        # Charge les paramètres OpenMax ajustés (MAVs + Weibull)
        params_path = self.config.get("openmax_params_path")
        if not params_path or not Path(params_path).exists():
            return

        import pickle
        with open(params_path, "rb") as f:
            self.openmax_params = pickle.load(f)
        print(f"OpenMax params loaded: {params_path}")

    def _load_gallery(self):
        # Charge la galerie d'embeddings de drones connus pour l'attribution Siamese
        gallery_path = self.config.get("gallery_path")
        if not gallery_path or not Path(gallery_path).exists():
            return

        data = np.load(gallery_path, allow_pickle=True)
        self.gallery = {
            "embeddings": data["embeddings"],  # [num_classes, embedding_dim]
            "class_names": data["class_names"].tolist(),
        }
        print(f"Gallery loaded: {len(self.gallery['class_names'])} classes ({gallery_path})")

    def _load_gnn(self):
        # Charge le GNN pour l'investigation multi-segments par graphe
        weights = self.config.get("gnn_weights")
        if not weights or not Path(weights).exists():
            return

        from src.models.gnn import RFDroneGNN
        self.gnn = RFDroneGNN(
            in_dim=self.config.get("gnn_emb_dim", 128),
            hidden_dim=self.config.get("gnn_hidden_dim", 256),
            num_classes=self.num_classes,
        ).to(self.device)
        self.gnn.load_state_dict(
            torch.load(weights, weights_only=True, map_location=self.device)
        )
        self.gnn.eval()
        print(f"GNN loaded: {weights}")

    def _load_explainability(self):
        # Charge GradCAM ou AttentionRollout selon le type de classificateur
        if self.classifier is None:
            return
        model_name = self.config.get("classifier_model", "resnet")
        if model_name == "cnn1d":
            return  # forensic pipeline works on spectrograms; skip raw-1D explainer
        try:
            from src.evaluation.explainability import (
                GradCAM, AttentionRollout, get_target_layer,
            )
            if model_name in ("ast", "transformer"):
                self.explainer = AttentionRollout(self.classifier, model_name=model_name)
                self.explainer_type = "attention_rollout"
            else:
                target_layer = get_target_layer(self.classifier, model_name)
                self.explainer = GradCAM(self.classifier, target_layer)
                self.explainer_type = "gradcam"
            print(f"Explainability: {self.explainer_type} ({model_name})")
        except Exception as e:
            print(f"WARNING: Explainability setup failed ({e}) — skipping")

    def _explain_segment(self, x, segment_id, predicted_class, output_dir):
        # Génère la carte Grad-CAM / attention pour un segment et sauvegarde le PNG.
        # Retourne le chemin du fichier sauvegardé, ou None en cas d'échec.
        try:
            from src.evaluation.explainability import plot_gradcam
            heatmap, _, conf = self.explainer.generate(x.clone(), target_class=predicted_class)

            class_label = (
                self.class_names[predicted_class]
                if predicted_class < len(self.class_names)
                else f"class_{predicted_class}"
            )
            fname = f"segment_{segment_id:04d}_{class_label.replace(' ', '_')}.png"
            save_path = Path(output_dir) / "explainability" / fname
            save_path.parent.mkdir(parents=True, exist_ok=True)

            spec_np = x.squeeze().cpu().detach().numpy()
            plot_gradcam(
                spec_np, heatmap, predicted_class, conf,
                class_names=self.class_names,
                output_path=str(save_path),
                title=f"Segment {segment_id} — {self.explainer_type}",
            )
            return str(save_path)
        except Exception as e:
            return None

    def analyze_segment(self, spectrogram_tensor):
        # Analyse complète d'un segment spectrogramme [1, 1, H, W]
        result = {}

        # 1. Classification supervisée
        if self.classifier is not None:
            with torch.no_grad():
                logits = self.classifier(spectrogram_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred].item()
                energy = torch.logsumexp(logits, dim=1).item()

            result["classification"] = {
                "predicted_class": int(pred),
                "predicted_label": (
                    self.class_names[pred] if pred < len(self.class_names)
                    else f"Unknown ({pred})"
                ),
                "confidence": round(float(confidence), 4),
                "energy_score": round(float(energy), 4),
                "probabilities": {
                    self.class_names[j] if j < len(self.class_names) else f"class_{j}":
                    round(float(probs[0, j]), 4)
                    for j in range(probs.shape[1])
                },
                "is_anomalous": confidence < self.anomaly_threshold,
            }

        # 2. OpenMax (détection open-set)
        if self.classifier is not None and self.openmax_params is not None:
            try:
                mavs = self.openmax_params["mavs"]
                weibull_params = self.openmax_params["weibull_params"]

                with torch.no_grad():
                    emb = self.classifier.get_embedding(spectrogram_tensor).cpu().numpy()[0]
                    logit = self.classifier(spectrogram_tensor).cpu().numpy()[0]

                # Calcul OpenMax pour un échantillon
                from scipy.stats import exponweib
                distances = np.array([
                    np.linalg.norm(emb - mavs[c]) for c in range(self.num_classes)
                ])
                ranked = np.argsort(distances)
                revised = logit.copy()
                unknown_score = 0.0

                for c in ranked[:3]:  # alpha=3
                    dist = distances[c]
                    try:
                        w_cdf = exponweib.cdf(dist, *weibull_params[c])
                    except Exception:
                        w_cdf = 0.0
                    reduction = logit[c] * w_cdf
                    revised[c] -= reduction
                    unknown_score += reduction

                revised_all = np.append(revised, unknown_score)
                exp_l = np.exp(revised_all - revised_all.max())
                openmax_probs = exp_l / exp_l.sum()

                result["openset"] = {
                    "unknown_probability": round(float(openmax_probs[-1]), 4),
                    "is_unknown": bool(openmax_probs[-1] > 0.5),
                    "known_class_probs": {
                        self.class_names[j] if j < len(self.class_names) else f"class_{j}":
                        round(float(openmax_probs[j]), 4)
                        for j in range(self.num_classes)
                    },
                }
            except Exception as e:
                result["openset"] = {"error": str(e)}

        # 3. Score d'anomalie VAE
        if self.vae is not None:
            with torch.no_grad():
                anomaly_scores = self.vae.anomaly_score(spectrogram_tensor)
                recon_error = anomaly_scores[0].item()

            result["anomaly"] = {
                "reconstruction_error": round(float(recon_error), 6),
                "is_anomalous": bool(recon_error > self.config.get("vae_threshold", 0.1)),
            }

        # 4. Attribution Siamese
        if self.siamese is not None and self.gallery is not None:
            with torch.no_grad():
                query_emb = self.siamese.get_embedding(spectrogram_tensor).cpu().numpy()
                gallery_embs = self.gallery["embeddings"]

                # Similarité cosinus avec chaque classe de la galerie
                similarities = np.dot(query_emb, gallery_embs.T).flatten()
                best_idx = int(np.argmax(similarities))

            result["attribution"] = {
                "most_similar_class": self.gallery["class_names"][best_idx],
                "similarity_score": round(float(similarities[best_idx]), 4),
                "all_similarities": {
                    name: round(float(sim), 4)
                    for name, sim in zip(self.gallery["class_names"], similarities)
                },
            }

        return result

    def analyze_file(self, file_path, output_dir=None, window_size=131072, hop_size=65536,
                     fs=1.0, nperseg=512, noverlap=256):
        # Analyse forensique complète d'un fichier signal, retourne la timeline par segment.
        # output_dir : si fourni, les heatmaps d'explicabilité sont sauvegardées ici.
        signal = load_dronerf_csv(file_path)
        segments = segment_signal(signal, window_size=window_size, hop_size=hop_size)

        explain_mode = self.config.get("explain_segments", "anomalous")

        timeline = []
        segment_embeddings = []  # pour la construction du graphe GNN

        for i, seg in enumerate(segments):
            start_sample = i * hop_size
            end_sample = start_sample + window_size

            # Calculer le spectrogramme
            _, _, S_log = compute_log_spectrogram(seg, fs=fs, nperseg=nperseg, noverlap=noverlap)
            x = torch.tensor(S_log, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

            # Analyser le segment (classification, OpenMax, VAE, Siamese)
            analysis = self.analyze_segment(x)

            # Collecter l'embedding pour le GNN
            if self.classifier is not None:
                with torch.no_grad():
                    emb = self.classifier.get_embedding(x)
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                    segment_embeddings.append(emb.cpu())

            entry = {
                "segment_id": i,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_time_s": start_sample / fs if fs > 0 else start_sample,
                "end_time_s": end_sample / fs if fs > 0 else end_sample,
                **analysis,
            }

            # 6. Explainability — génération sélective de la carte Grad-CAM / attention
            if self.explainer is not None and output_dir and explain_mode != "none":
                cls_result = analysis.get("classification", {})
                pred_class = cls_result.get("predicted_class", 0)
                should_explain = False

                if explain_mode == "all":
                    should_explain = True
                elif explain_mode == "drone":
                    should_explain = pred_class != 0
                else:  # "anomalous" (default)
                    is_anomalous = cls_result.get("is_anomalous", False)
                    is_unknown = analysis.get("openset", {}).get("is_unknown", False)
                    should_explain = is_anomalous or is_unknown

                if should_explain:
                    heatmap_path = self._explain_segment(x, i, pred_class, output_dir)
                    if heatmap_path:
                        entry["explainability"] = {
                            "method": self.explainer_type,
                            "heatmap_path": heatmap_path,
                        }

            timeline.append(entry)

        # 7. GNN : analyse multi-segments par graphe
        if self.gnn is not None and len(segment_embeddings) > 1:
            timeline = self._apply_gnn(timeline, segment_embeddings)

        return timeline

    def _apply_gnn(self, timeline, segment_embeddings):
        # Exécute le GNN sur les embeddings pour produire des prédictions raffinées par graphe
        from src.models.gnn import build_similarity_graph

        emb_tensor = torch.cat(segment_embeddings, dim=0)  # [N, D]
        adj = build_similarity_graph(
            emb_tensor,
            threshold=self.config.get("gnn_threshold", 0.5),
            k=self.config.get("gnn_k", 5),
        ).to(self.device)
        emb_tensor = emb_tensor.to(self.device)

        with torch.no_grad():
            logits = self.gnn(emb_tensor, adj)  # [N, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()

        for i, entry in enumerate(timeline):
            pred = int(preds[i])
            entry["gnn"] = {
                "predicted_class": pred,
                "predicted_label": (
                    self.class_names[pred] if pred < len(self.class_names)
                    else f"class_{pred}"
                ),
                "confidence": round(float(probs[i, pred]), 4),
                "probabilities": {
                    self.class_names[j] if j < len(self.class_names) else f"class_{j}":
                    round(float(probs[i, j]), 4)
                    for j in range(probs.shape[1])
                },
            }
        return timeline

    def generate_report(self, timeline, file_path, output_dir):
        # Génère un rapport forensique complet à partir de la timeline
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(timeline)

        # Résumé de classification
        class_dist = {}
        confidences = []
        anomalous_count = 0
        drone_count = 0

        for entry in timeline:
            if "classification" in entry:
                cls = entry["classification"]
                label = cls["predicted_label"]
                class_dist[label] = class_dist.get(label, 0) + 1
                confidences.append(cls["confidence"])
                if cls["is_anomalous"]:
                    anomalous_count += 1
                if cls["predicted_class"] != 0:
                    drone_count += 1

        # Résumé open-set
        unknown_segments = sum(
            1 for e in timeline
            if "openset" in e and isinstance(e["openset"], dict)
            and e["openset"].get("is_unknown", False)
        )

        # Résumé anomalies VAE
        vae_anomalies = sum(
            1 for e in timeline
            if "anomaly" in e and e["anomaly"].get("is_anomalous", False)
        )
        vae_scores = [
            e["anomaly"]["reconstruction_error"]
            for e in timeline if "anomaly" in e and "reconstruction_error" in e["anomaly"]
        ]

        # Résumé d'attribution
        attribution_counts = {}
        for e in timeline:
            if "attribution" in e:
                attr_class = e["attribution"]["most_similar_class"]
                attribution_counts[attr_class] = attribution_counts.get(attr_class, 0) + 1

        # Résumé GNN
        gnn_entries = [e["gnn"] for e in timeline if "gnn" in e]
        gnn_confidences = [e["confidence"] for e in gnn_entries]
        gnn_class_dist = {}
        for gnn_e in gnn_entries:
            label = gnn_e["predicted_label"]
            gnn_class_dist[label] = gnn_class_dist.get(label, 0) + 1
        gnn_drone_count = sum(1 for e in gnn_entries if e["predicted_class"] != 0)

        # Résumé explicabilité
        explained_entries = [e for e in timeline if "explainability" in e]
        explainability_summary = {
            "method": self.explainer_type,
            "segments_explained": len(explained_entries),
            "explain_mode": self.config.get("explain_segments", "anomalous"),
            "heatmap_paths": [
                e["explainability"]["heatmap_path"] for e in explained_entries
            ],
        } if explained_entries else {
            "method": self.explainer_type,
            "segments_explained": 0,
            "explain_mode": self.config.get("explain_segments", "anomalous"),
        }

        # Construire le rapport
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "source_file": str(file_path),
                "total_segments": total,
                "framework": "RF Drone Forensics — Integrated Pipeline",
                "components": {
                    "classifier": self.classifier is not None,
                    "openmax": self.openmax_params is not None,
                    "vae": self.vae is not None,
                    "siamese": self.siamese is not None and self.gallery is not None,
                    "gnn": self.gnn is not None,
                    "explainability": self.explainer is not None,
                },
                "config": {
                    k: str(v) for k, v in self.config.items()
                    if k not in ("device",)
                },
            },
            "classification_summary": {
                "class_distribution": class_dist,
                "drone_segments": drone_count,
                "anomalous_segments": anomalous_count,
                "average_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
            },
            "openset_summary": {
                "unknown_segments": unknown_segments,
                "unknown_rate": round(unknown_segments / max(total, 1) * 100, 2),
            },
            "anomaly_summary": {
                "vae_anomalous_segments": vae_anomalies,
                "mean_reconstruction_error": round(float(np.mean(vae_scores)), 6) if vae_scores else None,
            },
            "attribution_summary": {
                "most_attributed_class": max(attribution_counts, key=attribution_counts.get) if attribution_counts else None,
                "attribution_distribution": attribution_counts,
            },
            "gnn_summary": {
                "segments_analyzed": len(gnn_entries),
                "class_distribution": gnn_class_dist,
                "drone_segments": gnn_drone_count,
                "average_confidence": round(float(np.mean(gnn_confidences)), 4) if gnn_confidences else None,
            },
            "explainability_summary": explainability_summary,
            "timeline": timeline,
        }

        # Sauvegarder le rapport JSON
        report_path = out_dir / "integrated_forensic_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Integrated forensic report saved: {report_path}")

        # Tracer la timeline
        self._plot_integrated_timeline(timeline, out_dir / "integrated_timeline.png", file_path)

        return report

    def _plot_integrated_timeline(self, timeline, output_path, title=""):
        # Trace la timeline forensique intégrée multi-panneaux
        n_segments = len(timeline)
        segments = list(range(n_segments))

        # Déterminer le nombre de panneaux selon les composants disponibles
        panels = []
        if any("classification" in e for e in timeline):
            panels.append("classification")
        if any("openset" in e and "unknown_probability" in e.get("openset", {}) for e in timeline):
            panels.append("openset")
        if any("anomaly" in e for e in timeline):
            panels.append("anomaly")
        if any("attribution" in e for e in timeline):
            panels.append("attribution")
        if any("gnn" in e for e in timeline):
            panels.append("gnn")

        if not panels:
            return

        fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3 * len(panels)), sharex=True)
        if len(panels) == 1:
            axes = [axes]

        for ax, panel_type in zip(axes, panels):
            if panel_type == "classification":
                classes = [e.get("classification", {}).get("predicted_class", 0) for e in timeline]
                confidences = [e.get("classification", {}).get("confidence", 0) for e in timeline]
                colors = ["red" if c > 0 else "blue" for c in classes]
                ax.bar(segments, confidences, color=colors, alpha=0.7)
                ax.set_ylabel("Confidence")
                ax.set_title("Classification (red=drone, blue=background)")
                ax.axhline(y=self.anomaly_threshold, color="orange", linestyle="--", alpha=0.5)

            elif panel_type == "openset":
                unknown_probs = [
                    e.get("openset", {}).get("unknown_probability", 0) for e in timeline
                ]
                ax.fill_between(segments, unknown_probs, alpha=0.4, color="purple")
                ax.plot(segments, unknown_probs, "purple", linewidth=1)
                ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Unknown threshold")
                ax.set_ylabel("P(Unknown)")
                ax.set_title("Open-Set Detection (OpenMax)")
                ax.legend()

            elif panel_type == "anomaly":
                recon_errors = [
                    e.get("anomaly", {}).get("reconstruction_error", 0) for e in timeline
                ]
                ax.fill_between(segments, recon_errors, alpha=0.4, color="red")
                ax.plot(segments, recon_errors, "red", linewidth=1)
                ax.set_ylabel("Recon. Error")
                ax.set_title("VAE Anomaly Detection")

            elif panel_type == "attribution":
                # Classes attribuées uniques et couleurs
                all_attrs = [e.get("attribution", {}).get("most_similar_class", "?") for e in timeline]
                unique_classes = sorted(set(all_attrs))
                class_to_idx = {c: i for i, c in enumerate(unique_classes)}
                attr_indices = [class_to_idx[a] for a in all_attrs]
                sims = [e.get("attribution", {}).get("similarity_score", 0) for e in timeline]

                ax.bar(segments, sims, color=[plt.cm.tab10(idx % 10) for idx in attr_indices], alpha=0.7)
                ax.set_ylabel("Similarity")
                ax.set_title("Siamese Attribution")

                # Légende
                from matplotlib.patches import Patch
                legend_patches = [
                    Patch(facecolor=plt.cm.tab10(class_to_idx[c] % 10), label=c)
                    for c in unique_classes
                ]
                ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

            elif panel_type == "gnn":
                classes = [e.get("gnn", {}).get("predicted_class", 0) for e in timeline]
                confidences = [e.get("gnn", {}).get("confidence", 0) for e in timeline]
                colors = ["red" if c > 0 else "blue" for c in classes]
                ax.bar(segments, confidences, color=colors, alpha=0.7)
                ax.set_ylabel("Confidence")
                ax.set_title("GNN Graph-Refined Classification (red=drone, blue=background)")
                ax.axhline(y=self.anomaly_threshold, color="orange", linestyle="--", alpha=0.5)

        axes[-1].set_xlabel("Segment Index")
        fig.suptitle(f"Integrated Forensic Timeline — {Path(title).stem if title else ''}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Integrated timeline saved: {output_path}")

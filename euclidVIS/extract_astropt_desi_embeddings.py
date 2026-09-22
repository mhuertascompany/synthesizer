"""Extract DESI-only CLS embeddings from the supplied multimodal AstroPT model."""

import argparse
import csv
import inspect
import sys
from pathlib import Path

import numpy as np


MODALITY = "DESISpectrum"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--astropt-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(checkpoint_path, astropt_source, device):
    """Load public AstroPT layers and checkpoint-only CLS/modality parameters."""
    source_dir = astropt_source / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"No astroPT src directory at {source_dir}")
    sys.path.insert(0, str(source_dir))
    import torch
    from astropt.model import GPT, GPTConfig

    # This checkpoint is user-supplied and trusted. weights_only=False is
    # required for its pickled ModalityRegistry configuration object.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    registry = checkpoint["modality_registry"]
    if registry.get_config(MODALITY).input_size != 10:
        raise ValueError("Checkpoint DESISpectrum token width is not 10")
    if not config.get("use_cls_token") or config.get("cls_position") != "last":
        raise ValueError("Loader expects the trained CLS token in the last position")
    if config.get("spectra_size") != 7781 or config.get("spectra_patch_size") != 10:
        raise ValueError("Checkpoint spectral layout differs from 7781 samples / patch 10")
    if config.get("spectra_norm_type") != "asinh":
        raise ValueError("Prepared inputs assume the checkpoint's asinh normalization")

    allowed = set(inspect.signature(GPTConfig).parameters)
    config_args = {key: value for key, value in config.items() if key in allowed}
    config_args["tokeniser"] = registry.get_config(MODALITY).encoder_type
    model = GPT(GPTConfig(**config_args), registry, master_process=False)

    stripped = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model"].items()
    }
    cls_token = stripped.get("embedding_layer.cls_token")
    modality_embedding = stripped.get(f"embedding_layer.modality_embs.{MODALITY}")
    if cls_token is None or modality_embedding is None:
        raise KeyError("Checkpoint lacks the expected CLS or DESI modality embedding")
    base_state = {
        key: value for key, value in stripped.items()
        if not key.startswith("embedding_layer.")
    }
    incompatible = model.load_state_dict(base_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Public victor_branch does not match checkpoint base layers: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval().to(device)
    return model, cls_token.to(device), modality_embedding.to(device), config


def cls_embedding(model, tokens, positions, cls_token, modality_embedding):
    """Run the causal DESI sequence and return its last-position CLS state."""
    import torch

    x = model.encoders[MODALITY](tokens)
    x = x + model.embedders[MODALITY](positions)
    x = x + modality_embedding.to(dtype=x.dtype)
    cls = cls_token.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
    x = torch.cat((x, cls), dim=1)
    for block in model.transformer.h:
        x = block(x)
    return model.transformer.ln_f(x)[:, -1, :]


def main():
    args = parse_args()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a small test")
    model, cls_token, modality_embedding, checkpoint_config = load_model(
        args.checkpoint, args.astropt_source, args.device
    )
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Prepared-input manifest is empty")

    all_embeddings = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        arrays = [np.load(row["prepared"])["tokens"] for row in batch_rows]
        tokens = torch.from_numpy(np.stack(arrays)).to(args.device)
        positions = torch.arange(tokens.shape[1], device=args.device).expand(
            tokens.shape[0], -1
        )
        with torch.inference_mode():
            device_type = "cuda" if args.device.startswith("cuda") else "cpu"
            enabled = device_type == "cuda"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled):
                embedding = cls_embedding(
                    model, tokens, positions, cls_token, modality_embedding
                )
        all_embeddings.append(embedding.float().cpu().numpy())
        print(f"Embedded {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    embeddings = np.concatenate(all_embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        prepared=np.asarray([row["prepared"] for row in rows]),
        subhalo=np.asarray([int(float(row["subhalo"])) for row in rows]),
        snapshot=np.asarray([int(float(row["snapshot"])) for row in rows]),
        redshift=np.asarray([float(row["redshift"]) for row in rows]),
        mass=np.asarray([float(row["mass"]) for row in rows]),
        checkpoint=str(args.checkpoint.resolve()),
        layer="final_cls",
        modality=MODALITY,
        normalization=checkpoint_config["spectra_norm_type"],
    )
    print(f"Wrote {embeddings.shape} DESI embeddings to {args.output}")


if __name__ == "__main__":
    main()

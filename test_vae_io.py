"""Test VAE on rollout observations and save inputs/outputs."""
import argparse
from os import makedirs
from os.path import exists, join

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

from data.loaders import RolloutObservationDataset
from models.vae import VAE
from utils.misc import LSIZE, RED_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="VAE input/output tester")
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Experiment directory containing vae/best.tar",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/carracing",
        help="Rollout dataset root",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of test samples to save",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 avoids shared-buffer issues)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save test images (default: <logdir>/vae/test_io)",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="Dataset split to evaluate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = join(args.logdir, "vae", "best.tar")
    if not exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = args.output_dir or join(args.logdir, "vae", "test_io")
    inputs_dir = join(out_dir, "inputs")
    recons_dir = join(out_dir, "reconstructions")
    pairs_dir = join(out_dir, "pairs")
    makedirs(inputs_dir, exist_ok=True)
    makedirs(recons_dir, exist_ok=True)
    makedirs(pairs_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.ToTensor(),
        ]
    )

    is_train = args.split == "train"
    dataset = RolloutObservationDataset(args.data_root, transform, train=is_train)
    dataset.load_next_buffer()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = VAE(3, LSIZE).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    saved = 0
    latent_mu_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, mu, _ = model(batch)

            batch_size = batch.size(0)
            for i in range(batch_size):
                if saved >= args.num_samples:
                    break

                inp = batch[i].cpu()
                out = recon[i].cpu()
                pair = torch.cat([inp, out], dim=2)

                save_image(inp, join(inputs_dir, f"input_{saved:04d}.png"))
                save_image(out, join(recons_dir, f"recon_{saved:04d}.png"))
                save_image(pair, join(pairs_dir, f"pair_{saved:04d}.png"))

                latent_mu_all.append(mu[i].cpu().numpy())
                saved += 1

            if saved >= args.num_samples:
                break

    latent_mu = np.stack(latent_mu_all, axis=0) if latent_mu_all else np.zeros((0, LSIZE))
    np.save(join(out_dir, "latent_mu.npy"), latent_mu)

    with open(join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"checkpoint={ckpt_path}\n")
        f.write(f"split={args.split}\n")
        f.write(f"saved_samples={saved}\n")
        f.write(f"device={device}\n")
        f.write(f"latent_mu_shape={tuple(latent_mu.shape)}\n")

    print(f"Saved {saved} samples to: {out_dir}")


if __name__ == "__main__":
    main()

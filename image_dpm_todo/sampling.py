import argparse

import numpy as np
import torch
from dataset import tensor_to_pil_image
from model import DiffusionModule
from scheduler import DPMSolverScheduler
from pathlib import Path


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    dpm = DiffusionModule(None, None)
    dpm.load(args.ckpt_path)
    dpm.eval()
    dpm = dpm.to(device)

    num_train_timesteps = dpm.var_scheduler.num_train_timesteps
    dpm.var_scheduler = DPMSolverScheduler(num_train_timesteps, 1e-4, 0.02, "linear").to(device)

    # to enable high-order sampling
    dpm.var_scheduler.net_forward_fn = dpm.network.forward

    total_num_samples = 500
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        # Sample with CFG.
        assert dpm.network.use_cfg, f"The model was not trained to support CFG."
        samples = dpm.sample(
            (B,3,dpm.image_resolution, dpm.image_resolution),
            num_inference_timesteps=20,
            order=args.dpm_solver_order,
            class_label=torch.randint(1, 4, (B,)),
            guidance_scale=args.cfg_scale,
        )

        pil_images = tensor_to_pil_image(samples)

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--dpm_solver_order", type=int, default=1, choices=[1,2])

    args = parser.parse_args()
    main(args)

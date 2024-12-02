import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from Models.imputers.utils import training_loss, calc_diffusion_hyperparams, sampling


class Trainer(object):
    def __init__(self, config, args, model, dataloader):
        super().__init__()
        self.model = model
        self.device = args.device
        self.train_num_steps = config["solver"]["n_iters"]
        self.only_generate_missing = config["solver"]["only_generate_missing"]
        self.dl = dataloader["dataloader"]
        self.config = config
        self.args = args

        self.results_folder = Path(
            f"{args.save_dir}/{config['solver']['results_folder']}"
        )
        os.makedirs(self.results_folder, exist_ok=True)

        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=config["solver"]["learning_rate"]
        )

        self.diffusion_hyperparams = calc_diffusion_hyperparams(
            **config["diffusion_config"]
        )

        for key in self.diffusion_hyperparams:
            if key != "T":
                self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].to(
                    self.device
                )

    def save(self):
        save_dir = f"{self.results_folder}/checkpoint_{self.args.client_id}.pt"
        data = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(data, save_dir)

    def load(self):
        save_dir = f"{self.results_folder}/checkpoint_{self.args.client_id}.pt"
        data = torch.load(save_dir, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.opt.load_state_dict(data["opt"])

    def train(self):
        total_losses = 0.0
        for _ in range(self.train_num_steps):
            total_loss = 0.0
            for batch in self.dl:
                batch = batch.to(self.device)
                batch = batch.permute(0, 2, 1)
                mask = torch.zeros_like(batch)
                loss_mask = ~mask.bool()

                assert batch.size() == mask.size() == loss_mask.size()

                self.opt.zero_grad()
                X = batch, batch, mask, loss_mask
                loss = training_loss(
                    self.model,
                    nn.MSELoss(),
                    X,
                    self.diffusion_hyperparams,
                    only_generate_missing=self.only_generate_missing,
                )

                loss.backward()
                self.opt.step()
                total_loss += loss.item()

            total_losses += total_loss

        print("training complete")
        return total_losses / self.train_num_steps

    def sample(self, num, size_every, shape):
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            # Unconditional sampling
            sample_size = (size_every, shape[1], shape[0])
            dummy = torch.zeros(sample_size).to(self.device)
            sample = sampling(
                self.model,
                sample_size,
                self.diffusion_hyperparams,
                cond=dummy,
                mask=dummy,
                only_generate_missing=self.only_generate_missing,
            ).permute(0, 2, 1)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples

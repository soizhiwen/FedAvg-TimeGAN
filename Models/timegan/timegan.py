"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen(robinbg@foxmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from Models.timegan.model import Encoder, Recovery, Generator, Discriminator, Supervisor


def cycle(dl):
    while True:
        for data in dl:
            yield data


class BaseModel:
    """Base Model for timegan"""

    def __init__(self, config, args, dataloader):
        self.config = config
        self.args = args
        self.device = args.device
        self.train_num_steps = config["solver"]["n_iters"]
        self.lr = config["solver"]["learning_rate"]
        self.w_gamma = config["solver"]["w_gamma"]
        self.w_g = config["solver"]["w_g"]
        self.dl = cycle(dataloader["dataloader"]) if dataloader else None
        self.results_folder = Path(
            f"{args.save_dir}/{config['solver']['results_folder']}"
        )
        os.makedirs(self.results_folder, exist_ok=True)

    def save(self):
        save_dir = f"{self.results_folder}/checkpoint_{self.args.client_id}.pt"
        data = {
            "nete": self.nete.state_dict(),
            "netr": self.netr.state_dict(),
            "netg": self.netg.state_dict(),
            "netd": self.netd.state_dict(),
            "nets": self.nets.state_dict(),
        }
        torch.save(data, save_dir)

    def train_one_iter_er(self):
        """Train the model for one epoch."""

        self.nete.train()
        self.netr.train()

        self.X = next(self.dl).to(self.device)

        # train encoder & decoder
        self.optimize_params_er()

    def train_one_iter_er_(self):
        """Train the model for one epoch."""

        self.nete.train()
        self.netr.train()

        self.X = next(self.dl).to(self.device)

        # train encoder & decoder
        self.optimize_params_er_()

    def train_one_iter_s(self):
        """Train the model for one epoch."""

        # self.nete.eval()
        self.nets.train()

        self.X = next(self.dl).to(self.device)

        # train superviser
        self.optimize_params_s()

    def train_one_iter_g(self):
        """Train the model for one epoch."""

        """self.netr.eval()
    self.nets.eval()
    self.netd.eval()"""
        self.netg.train()

        self.X = next(self.dl).to(self.device)
        self.Z = torch.rand_like(self.X)

        # train superviser
        self.optimize_params_g()

    def train_one_iter_d(self):
        """Train the model for one epoch."""
        """self.nete.eval()
    self.netr.eval()
    self.nets.eval()
    self.netg.eval()"""
        self.netd.train()

        self.X = next(self.dl).to(self.device)
        self.Z = torch.rand_like(self.X)

        # train superviser
        self.optimize_params_d()

    def train(self):
        """Train the model"""
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()

        for _ in range(self.train_num_steps):
            # Train for one iter
            self.train_one_iter_er()

        print("Encoder training done")

        for _ in range(self.train_num_steps):
            # Train for one iter
            self.train_one_iter_s()

        print("Superviser training done")

        for _ in range(self.train_num_steps):
            # Train for one iter
            for _ in range(2):
                self.train_one_iter_g()
                self.train_one_iter_er_()

            self.train_one_iter_d()

        print("All training done")

    def sample(self, num, size_every, shape):
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample_size = (size_every, shape[0], shape[1])
            self.Z = torch.randn(sample_size, device=self.device)
            self.E_hat = self.netg(self.Z)  # [?, 24, 24]
            self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]
            sample = self.netr(self.H_hat)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples


class TimeGAN(BaseModel):
    def __init__(self, config, args, dataloader):
        super(TimeGAN, self).__init__(config, args, dataloader)

        # Create and initialize networks.
        self.nete = Encoder(self.config["model"]).to(self.device)
        self.netr = Recovery(self.config["model"]).to(self.device)
        self.netg = Generator(self.config["model"]).to(self.device)
        self.netd = Discriminator(self.config["model"]).to(self.device)
        self.nets = Supervisor(self.config["model"]).to(self.device)

        # loss
        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()

        # Setup optimizer
        self.optimizer_e = optim.Adam(
            self.nete.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        self.optimizer_r = optim.Adam(
            self.netr.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        self.optimizer_g = optim.Adam(
            self.netg.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.netd.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        self.optimizer_s = optim.Adam(
            self.nets.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

    def forward_e(self):
        """Forward propagate through netE"""
        self.H = self.nete(self.X)

    def forward_er(self):
        """Forward propagate through netR"""
        self.H = self.nete(self.X)
        self.X_tilde = self.netr(self.H)

    def forward_g(self):
        """Forward propagate through netG"""
        self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        self.E_hat = self.netg(self.Z)

    def forward_dg(self):
        """Forward propagate through netD"""
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
        """Forward propagate through netG"""
        self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
        """Forward propagate through netS"""
        self.H_supervise = self.nets(self.H)
        # print(self.H, self.H_supervise)

    def forward_sg(self):
        """Forward propagate through netS"""
        self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
        """Forward propagate through netD"""
        self.Y_real = self.netd(self.H)
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def backward_er(self):
        """Backpropagate through netE"""
        self.err_er = self.l_mse(self.X_tilde, self.X)
        self.err_er.backward(retain_graph=True)
        # print("Loss: ", self.err_er)

    def backward_er_(self):
        """Backpropagate through netE"""
        self.err_er_ = self.l_mse(self.X_tilde, self.X)
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
        self.err_er.backward(retain_graph=True)

    #  print("Loss: ", self.err_er_, self.err_s)
    def backward_g(self):
        """Backpropagate through netG"""
        self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))

        self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
        self.err_g_V1 = torch.mean(
            torch.abs(
                torch.sqrt(torch.std(self.X_hat, [0])[1] + 1e-6)
                - torch.sqrt(torch.std(self.X, [0])[1] + 1e-6)
            )
        )  # |a^2 - b^2|
        self.err_g_V2 = torch.mean(
            torch.abs((torch.mean(self.X_hat, [0])[0]) - (torch.mean(self.X, [0])[0]))
        )  # |a - b|
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_g = (
            self.err_g_U
            + self.err_g_U_e * self.w_gamma
            + self.err_g_V1 * self.w_g
            + self.err_g_V2 * self.w_g
            + torch.sqrt(self.err_s)
        )
        self.err_g.backward(retain_graph=True)
        # print("Loss G: ", self.err_g)

    def backward_s(self):
        """Backpropagate through netS"""
        self.err_s = self.l_mse(self.H[:, 1:, :], self.H_supervise[:, :-1, :])
        self.err_s.backward(retain_graph=True)
        # print("Loss S: ", self.err_s)

    #   print(torch.autograd.grad(self.err_s, self.nets.parameters()))

    def backward_d(self):
        """Backpropagate through netD"""
        self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
        self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
        self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
        self.err_d = (
            self.err_d_real + self.err_d_fake + self.err_d_fake_e * self.w_gamma
        )
        if self.err_d > 0.15:
            self.err_d.backward(retain_graph=True)

    # print("Loss D: ", self.err_d)

    def optimize_params_er(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_er()

        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_er_(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_er()
        self.forward_s()
        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er_()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_s(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_s()

        # Backward-pass
        # nets
        self.optimizer_s.zero_grad()
        self.backward_s()
        self.optimizer_s.step()

    def optimize_params_g(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_s()
        self.forward_g()
        self.forward_sg()
        self.forward_rg()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
        self.optimizer_s.step()

    def optimize_params_d(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_g()
        self.forward_sg()
        self.forward_d()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

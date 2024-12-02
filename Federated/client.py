import os
import copy
import time
import warnings
import argparse
from typing import Any
from collections import OrderedDict

import torch
import numpy as np
import numpy.random as npr
import flwr as fl
from flwr.common import NDArrays, Scalar

from Models.timegan.timegan import TimeGAN
from Models.timegan.utils import unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader_fed
from Federated.utils import (
    load_data_partitions,
    write_csv,
    context_fid,
    cross_corr,
    discriminative,
    predictive,
)

warnings.filterwarnings("ignore")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config: dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

        self.save_dir = args.save_dir
        self.client_id = args.client_id

        # Load data
        data = load_data_partitions(
            path=config["dataloader"]["params"]["data_root"],
            num_clients=args.num_clients,
            partition_id=args.client_id,
            pub_ratio=args.pub_ratio,
            split_mode=args.split_mode,
        )

        # Shuffle data
        st0 = np.random.get_state()
        np.random.seed(args.client_id)
        window = config["dataloader"]["params"]["window"]
        data = data.reshape(-1, window, data.shape[-1])
        npr.shuffle(data)
        data = np.concatenate(data)
        np.random.set_state(st0)

        # Build dataloader
        self.dl_info = build_dataloader_fed(
            config=config,
            name=args.client_id,
            data=data,
            seed=args.client_id,
            save_dir=args.save_dir,
        )
        self.filename = (
            f"{self.dl_info['dataset'].name}_{self.dl_info['dataset'].window}"
        )

        self.trainer = TimeGAN(config=config, args=args, dataloader=self.dl_info)
        self.nete_params_len = len(self.trainer.nete.state_dict().values())
        self.netr_params_len = len(self.trainer.netr.state_dict().values())
        self.netg_params_len = len(self.trainer.netg.state_dict().values())
        self.netd_params_len = len(self.trainer.netd.state_dict().values())
        self.nets_params_len = len(self.trainer.nets.state_dict().values())

    def get_parameters(self):
        nete_params, netr_params, netg_params = [], [], []
        netd_params, nets_params = [], []
        param_bytes = 0

        for _, val in self.trainer.nete.state_dict().items():
            nete_params.append(val.cpu().numpy())
            param_bytes += nete_params[-1].nbytes

        for _, val in self.trainer.netr.state_dict().items():
            netr_params.append(val.cpu().numpy())
            param_bytes += netr_params[-1].nbytes

        for _, val in self.trainer.netg.state_dict().items():
            netg_params.append(val.cpu().numpy())
            param_bytes += netg_params[-1].nbytes

        for _, val in self.trainer.netd.state_dict().items():
            netd_params.append(val.cpu().numpy())
            param_bytes += netd_params[-1].nbytes

        for _, val in self.trainer.nets.state_dict().items():
            nets_params.append(val.cpu().numpy())
            param_bytes += nets_params[-1].nbytes

        parameters = nete_params + netr_params + netg_params + netd_params + nets_params

        return parameters, param_bytes

    def set_parameters(self, parameters: NDArrays) -> None:
        params = parameters[: self.nete_params_len]
        params_dict = zip(self.trainer.nete.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.nete.load_state_dict(state_dict, strict=True)
        parameters = parameters[self.nete_params_len :]

        params = parameters[: self.netr_params_len]
        params_dict = zip(self.trainer.netr.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.netr.load_state_dict(state_dict, strict=True)
        parameters = parameters[self.netr_params_len :]

        params = parameters[: self.netg_params_len]
        params_dict = zip(self.trainer.netg.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.netg.load_state_dict(state_dict, strict=True)
        parameters = parameters[self.netg_params_len :]

        params = parameters[: self.netd_params_len]
        params_dict = zip(self.trainer.netd.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.netd.load_state_dict(state_dict, strict=True)
        parameters = parameters[self.netd_params_len :]

        params = parameters[: self.nets_params_len]
        params_dict = zip(self.trainer.nets.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.nets.load_state_dict(state_dict, strict=True)

        param_bytes = 0
        for _, val in self.trainer.nete.state_dict().items():
            param_bytes += val.cpu().numpy().nbytes

        for _, val in self.trainer.netr.state_dict().items():
            param_bytes += val.cpu().numpy().nbytes

        for _, val in self.trainer.netg.state_dict().items():
            param_bytes += val.cpu().numpy().nbytes

        for _, val in self.trainer.netd.state_dict().items():
            param_bytes += val.cpu().numpy().nbytes

        for _, val in self.trainer.nets.state_dict().items():
            param_bytes += val.cpu().numpy().nbytes

        return param_bytes

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # Update local model parameters
        total_param_bytes = 0
        param_bytes = self.set_parameters(parameters)
        total_param_bytes += param_bytes

        server_round = config["server_round"]
        self.trainer.train_num_steps = config["local_epochs"]
        train_time = time.time()
        self.trainer.train()
        train_time = time.time() - train_time
        self.write_client_results({"train_time": train_time}, server_round)
        self.trainer.save()

        parameters_prime, param_bytes = self.get_parameters()
        total_param_bytes += param_bytes
        self.write_client_results({"param_bytes": total_param_bytes}, server_round)

        metrics = {"train_loss": 0.0}
        self.write_client_results(metrics, server_round)

        return parameters_prime, len(self.dl_info["dataset"]), metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        server_round = config["server_round"]

        if server_round == self.args.num_rounds:
            # Update local model parameters
            self.set_parameters(parameters)

            size_every = config["size_every"]
            metric_iterations = config["metric_iterations"]

            dataset = self.dl_info["dataset"]
            seq_len, feat_dim = dataset.window, dataset.var_num

            synth_time = time.time()
            synth = self.trainer.sample(
                len(dataset), size_every, shape=[seq_len, feat_dim]
            )
            synth_time = time.time() - synth_time
            self.write_client_results({"synth_time": synth_time}, server_round)

            if dataset.auto_norm:
                synth = unnormalize_to_zero_to_one(synth)

            # Save synthetic data
            synth_dir = f"{self.save_dir}/synthetic"
            os.makedirs(synth_dir, exist_ok=True)
            np.save(f"{synth_dir}/norm_{self.filename}.npy", synth)

            unnorm_synth = dataset.scaler.inverse_transform(
                synth.reshape(-1, feat_dim)
            ).reshape(synth.shape)
            np.save(f"{synth_dir}/{self.filename}.npy", unnorm_synth)

            # Compute metrics for all features
            metrics = {}
            real = np.load(f"{dataset.dir}/norm_{self.filename}_test.npy")

            try:
                all_ctx_fid = context_fid(real, synth, metric_iterations)
                metrics["all_context_fid"] = float(all_ctx_fid[0])
                metrics["all_context_fid_sigma"] = float(all_ctx_fid[1])
            except Exception as e:
                print(f"Error: {e}")

            try:
                all_cross_corr = cross_corr(real, synth, metric_iterations)
                metrics["all_cross_corr"] = float(all_cross_corr[0])
                metrics["all_cross_corr_sigma"] = float(all_cross_corr[1])
            except Exception as e:
                print(f"Error: {e}")

            try:
                all_discriminative = discriminative(real, synth, metric_iterations)
                metrics["all_discriminative"] = float(all_discriminative[0])
                metrics["all_discriminative_sigma"] = float(all_discriminative[1])
            except Exception as e:
                print(f"Error: {e}")

            try:
                all_predictive = predictive(real, synth, metric_iterations)
                metrics["all_predictive"] = float(all_predictive[0])
                metrics["all_predictive_sigma"] = float(all_predictive[1])
            except Exception as e:
                print(f"Error: {e}")

            self.write_client_results(metrics, server_round)
            return 0.0, len(real), metrics

        return 0.0, 1, {}

    def write_client_results(
        self,
        results: dict[str, float],
        server_round: int,
    ) -> None:
        """Write client results to disk."""
        for k, v in results.items():
            fields = [server_round, v, self.client_id]
            write_csv(fields, f"clients_{k}", self.save_dir)


def get_client_fn(config, args):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """
    config = copy.deepcopy(config)

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        args.client_id = int(cid)
        return FlowerClient(config, args).to_client()

    return client_fn

import torch
import numpy as np
from tqdm import tqdm
import os
import json

from .utils import find_best_checkpoint, generate_samples_and_run_eval
from ..utils import prepare_data_and_model

def evaluate_best_ensemble(config, match_string, data_parameters = None, accurate_ensemble_samples = None):
    experiment_path = f"/share/gpu0/asaoulis/cmd/checkpoints/{config.experiment_name}"
    results = {}
    
    model = EnsembleModel(config, experiment_path, match_string, data_parameters)

    metrics = generate_samples_and_run_eval(model, data_parameters[0][0], accurate_ensemble_samples)
    
    results[experiment_path] = {
        "best_checkpoints": model.checkpoints,
        "metrics": metrics
    }
    
    results_path = os.path.join(experiment_path, f"ensemble_evaluation_results_{match_string}.json")
    with open(results_path, "w") as f:
        json.dump(results[experiment_path], f, indent=4)
    
    return results

class EnsembleModel:
    def __init__(self, config, experiment_folder, match_string, data_parameters):
        self.models = []
        self.checkpoints = []
        self.load_models(config, experiment_folder, match_string, data_parameters)
        self.test_dataloader = self.models[0].test_dataloader

        self.posteriors = [model.build_posterior_object() for model in self.models]

    def load_models(self, config, experiment_folder, match_string, data_parameters):
        run_folders = [os.path.join(experiment_folder, d) for d in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, d))]
        for run_folder in run_folders:
            if match_string not in run_folder:
                continue
            best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
            self.checkpoints.append(best_checkpoint)
            config.checkpoint_path = best_checkpoint
            _, _, model, _ = prepare_data_and_model(config, data_parameters)
            model.load_from_checkpoint(best_checkpoint)
            model.to("cuda")
            model.eval()
            self.models.append(model)
    def mean_log_prob(self, x, y):
        """
        Computes the mean log probability across all models in the ensemble.
        
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
        
        Returns:
            torch.Tensor: The mean log probability.
        """
        log_probs = [model.model.log_prob(x, y) for model in self.models]
        return torch.stack(log_probs).mean(dim=0)
    
    def generate_samples(self, test_dataloader, num_samples=10000):
        """
        Generates samples from all models and combines them.
        
        Args:
            test_dataloader (DataLoader): The test dataloader.
            num_samples (int): Number of samples per model.
        
        Returns:
            np.ndarray: Combined ensemble samples.
        """
        test_y, test_x = test_dataloader.dataset.tensors
        test_y = test_y.to('cuda', dtype=torch.float32).unsqueeze(1)
        ensemble_samples = []
        num_examples = len(test_x)
        for i in tqdm(range(num_examples), desc="Sampling", total=num_examples):
            samples = []
            y = test_y[i]

            for posterior in self.posteriors:

                x_samples = posterior.sample((num_samples//len(self.models),), x=y, show_progress_bars=False)
                samples.append(x_samples)
        
            samples = torch.concatenate(samples, dim=0)
            ensemble_samples.append(samples)
        ensemble_samples = torch.stack(ensemble_samples).permute(1, 0, 2)
        return test_x, ensemble_samples

    def compute_avg_log_prob(self):
        """
        Computes the average log probability over the test dataset for the ensemble.
        
        Returns:
            float: The per-element average log probability across all models.
        """
        test_dataloader = self.models[0].test_dataloader
        transfer_fn = self.models[0].transfer_batch_to_device
        device = self.models[0].device
        
        all_log_probs = []
        for batch in test_dataloader:
            batch = transfer_fn(batch, device, 0)
            batch_log_probs = torch.stack([model.forward(batch[0], batch[1]) for model in self.models])
            avg_batch_log_prob = batch_log_probs.mean(dim=0)  # Average over models per element in batch
            all_log_probs.append(avg_batch_log_prob)
        
        avg_log_prob = torch.cat(all_log_probs, dim=0).mean().item()
        return avg_log_prob
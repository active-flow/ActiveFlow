import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import time
class ActivationModule:
    def __init__(self) :
        self.histograms_only: bool = False
        self.visualize_strategy: bool = True
        self.activations = None
        self.histograms = None
        self.file_path = None
        self.num_layers = 0
        self.num_weights = 0
    def set_visuable_strategy(self, s = True) :
        self.visualize_strategy = s
    def set_init_dict(self , num_layers , num_weights, file_path) :
        self.num_layers = num_layers
        self.num_weights = num_weights
        self.activations = [[[] for id in range(num_weights)] for _ in range(num_layers)]
        self.histograms= [[{} for id in range(num_weights)] for _ in range(num_layers)]
        self.file_path = file_path

    # TODO: optimize this function
    def fd_th(self, weight, sp) :
        tensor = weight.flatten().cpu().numpy()  
        threshold = np.percentile(tensor, q=sp*100)
        return threshold.item()        

    def fd_th_by_histogram(self, histogram, sp) :
        bin_centers = histogram["bin_centers"]
        counts = histogram["histogram"]

        
        abs_bin_centers = torch.abs(bin_centers)

        unique_bins, inverse_indices = torch.unique(abs_bin_centers, return_inverse=True)
        new_counts = torch.zeros_like(unique_bins)
        new_counts.scatter_add_(0, inverse_indices, counts)

        cumulative_counts = torch.cumsum(new_counts, dim=0)
        total_count = cumulative_counts[-1]

        target_count = sp * total_count
        idx = torch.searchsorted(cumulative_counts, target_count)

        if idx == 0:
            threshold = unique_bins[0]
        elif idx == len(unique_bins):
            threshold = unique_bins[-1]
        else:
            lower_count = cumulative_counts[idx - 1]
            upper_count = cumulative_counts[idx]
            lower_value = unique_bins[idx - 1]
            upper_value = unique_bins[idx]

            fraction = (target_count - lower_count) / (upper_count - lower_count)
            threshold = lower_value + fraction * (upper_value - lower_value)

        return threshold.item()

    def cal_histograms(self, activations):
        
        flattened = activations.flatten().detach().to('cuda')
        
        k_low = int(0.01 * flattened.numel())
        k_high = int(0.99 * flattened.numel())
        lower_bound = torch.kthvalue(flattened, k_low).values
        upper_bound = torch.kthvalue(flattened, k_high).values

        mask = (flattened >= lower_bound) & (flattened <= upper_bound)
        filtered = flattened[mask]

        bins = 1000
        hist = torch.histc(filtered, bins=bins, min=lower_bound.item(), max=upper_bound.item())
        bin_edges = torch.linspace(lower_bound.item(), upper_bound.item(), steps=bins+1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return hist.cpu().numpy(), bin_edges.cpu().numpy(), bin_centers.cpu().numpy()

    
    def visualize_histogram(self, layer_idx, weight_idx):
        histograms_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}", "histograms.pt")

        # load histogram
        histogram_data = torch.load(histograms_path, map_location='cpu')
        histogram, bin_edges, bin_centers = histogram_data['histogram'], histogram_data['bin_edges'], histogram_data['bin_centers']

        plt.figure(figsize=(12, 8))
        plt.bar(bin_centers, histogram, width=(bin_edges[-1] - bin_edges[0]) / len(bin_edges), alpha=0.7, color='blue', label='Histogram')
        plt.xlabel("Value", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title(f"Histogram for Layer {layer_idx}, Weight {weight_idx}", fontsize=16)

        # Calculate the absolute values and their frequencies
        absolute_values = np.abs(bin_centers)
        sorted_indices = np.argsort(absolute_values)
        sorted_absolute_values = absolute_values[sorted_indices]
        sorted_frequencies = histogram[sorted_indices]

        # Calculate cumulative frequency
        cumulative_frequency = np.cumsum(sorted_frequencies)
        total_frequency = cumulative_frequency[-1]
        percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        percentile_thresholds = []

        for p in percentiles:
            threshold_index = np.searchsorted(cumulative_frequency, p * total_frequency)
            if threshold_index < len(sorted_absolute_values):
                percentile_thresholds.append(sorted_absolute_values[threshold_index])
            else:
                percentile_thresholds.append(sorted_absolute_values[-1])

        # Plot vertical lines for each percentile threshold
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i, (p, threshold) in enumerate(zip(percentiles, percentile_thresholds)):
            plt.axvline(x=threshold, color=colors[i % len(colors)], linestyle='--', linewidth=2.0, label=f'{int(p*100)}%')
            plt.axvline(x=-threshold, color=colors[i % len(colors)], linestyle='--', linewidth=2.0)
            plt.text(threshold, plt.ylim()[1]*0.95 - i*0.04*plt.ylim()[1], f'{int(p*100)}%', color=colors[i % len(colors)], ha='center', fontsize=12)
            plt.text(-threshold, plt.ylim()[1]*0.95 - i*0.04*plt.ylim()[1], f'{int(p*100)}%', color=colors[i % len(colors)], ha='center', fontsize=12)

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # save histogram
        save_dir = os.path.join(self.file_path, "histograms_png")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"layer_{layer_idx}-weight_{weight_idx}-histograms.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()

    
    def save_layer_weight(self, layer_idx, weight_idx) :
        if not self.activations[layer_idx][weight_idx]:
            print('None', self.activations[layer_idx][weight_idx])
            return 
        else :   
            layer_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}")
            os.makedirs(layer_path, exist_ok=True)

            activations_path = os.path.join(layer_path, "activations.pt")
            combine_activations = torch.cat(self.activations[layer_idx][weight_idx], dim=0)
            if self.histograms_only == False :
                torch.save(combine_activations, activations_path)

            histograms_path = os.path.join(layer_path, "histograms.pt")
            histogram, bin_edges, bin_centers = self.cal_histograms(combine_activations)
            # print({'histogram': histogram, 'bin_edges': bin_edges, 'bin_centers': bin_centers})
            torch.save({'histogram': histogram, 'bin_edges': bin_edges, 'bin_centers': bin_centers}, histograms_path)

    def clear_layer_weight(self, layer_idx, weight_idx) :
        # print('finish', layer_idx, weight_idx)
        self.activations[layer_idx][weight_idx] = []
        
    def grab_activations(self, x, layer_idx, weight_idx):
        if x.size(1) > 1:  # Check if seq_len > 1
            # print('grab ', layer_idx, weight_idx, x.size())
            self.activations[layer_idx][weight_idx].append(x.detach().squeeze(0).cpu().float())
            
            self.save_layer_weight(layer_idx, weight_idx)
            self.clear_layer_weight(layer_idx, weight_idx)

            if self.visualize_strategy:
                start_time = time.time()
                self.visualize_histogram(layer_idx, weight_idx)
                visualize_time = time.time() - start_time  
                print(f"visualize_histogram ({layer_idx}, {weight_idx}) took {visualize_time:.6f} seconds")
    
    def save_activations(self):
        for layer_idx in range(self.num_layers):
            for weight_idx in range(self.num_weights) :
                self.save_layer_weight(layer_idx, weight_idx)
  
    def find_threshold(self, sp, output_path):
        thresholds = {}
        for layer_idx in range(self.num_layers):
            layer_thresholds = {}
            for weight_idx in range(self.num_weights):
                layer_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}")
                if self.histograms_only == False :
                    activations_path = os.path.join(layer_path, "activations.pt")
                    weight = torch.load(activations_path)
                    threshold = self.fd_th(torch.abs(weight), sp)
                else :
                    histograms_path = os.path.join(layer_path, "histograms.pt")
                    histograms = torch.load(histograms_path)
                    threshold = self.fd_th_by_histogram(histograms, sp)
                
                layer_thresholds[weight_idx]=threshold
                print('find_threshold:  ', layer_idx, weight_idx, threshold)                
                
                del weight
                
                
            thresholds[layer_idx]=layer_thresholds

        # Save thresholds to output_path
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"sparse-{sp}.json")
        with open(save_path,'w') as f:
            json.dump(thresholds, f)
        # print(thresholds)
        # torch.save(thresholds, )
        return thresholds
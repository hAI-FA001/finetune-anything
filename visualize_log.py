import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='semantic_seg', type=str)
parser.add_argument('--log_path', default=None, type=str)
parser.add_argument('--color', default='orange', type=str)
parser.add_argument('--log_scale', default=False, type=bool)

if __name__ == "__main__":
    args = parser.parse_args()
    task_name = args.task_name
    log_path = args.log_path
    color = args.color
    log_scale = args.log_scale

    if not log_path:
        log_path = f"./experiment/log/{task_name}/log_file.txt"

    loss_values = []
    losses = ['ce', 'mse', 'multi_label_soft_margin', 'test_custom']
    with open(log_path, "r") as f:
        which_loss = None
        for line in f:
            if which_loss is None:
                for loss in losses:
                    if loss in line:
                        which_loss = loss
                        break
            
            if which_loss in line:
                itera = line.split(f", {which_loss}")[0]
                itera = itera.split("iteration :")[1]
                itera = float(itera.strip())
                line = line.split(f"{which_loss} :")[1].split(", total")[0]
                loss_values.append((itera, float(line.strip())))


    loss_values = np.array(loss_values)
    print('Loss array: ', loss_values.shape)

    plt.figure()
    plt.title("SAM Validation Loss Curve")
    plt.plot(loss_values[:, 0], loss_values[:, 1], color=color)
    
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

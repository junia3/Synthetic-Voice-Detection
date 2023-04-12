import matplotlib.pyplot as plt
import os

def visualize_logs(logs: dict, exp_name: str) -> None:
    for key, value in logs.items():
        plt.figure(figsize=(10, 6))
        plt.title(f"{key} Graph")
        for k, v in value.items():
            plt.plot(v, label=f"{key} {k}")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(f"{key}")
        plt.grid(True)
        filename = os.path.join(f"checkpoint/{exp_name}", f'graph_{key}.png')
        plt.savefig(filename)
        plt.close()
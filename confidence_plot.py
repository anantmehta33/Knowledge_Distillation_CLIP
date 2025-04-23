import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

def main():

    conf_file = "alphas_epoch0.pkl"

    with open(conf_file, 'rb') as f:
        alphas = pickle.load(f)

    alphas_arr = np.array(alphas).squeeze().flatten()

    print(f"Alphas shape: {alphas_arr.shape}, first element: {alphas_arr}")

    plt.hist(alphas_arr, bins=100, edgecolor='black')
    plt.yscale('log')
    plt.title("Distribution of Confidence Values")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig("confidence_distribution_teacher.png", dpi=300, bbox_inches='tight')  # Save as high-res PNG
    plt.close()  # Close the figure to avoid display in notebooks or scripts

if __name__ == "__main__":
    main()
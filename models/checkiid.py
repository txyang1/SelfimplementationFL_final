from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def check_iid(dataset_loader):
    label_counts = Counter()
    
    # Iterate through DataLoader to count labels
    for _, labels in dataset_loader:
        label_counts.update(labels.numpy())

    # Print label counts
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

    # Plot label distribution
    labels, counts = zip(*label_counts.items())
    plt.bar(labels, counts)
    plt.xlabel('Labels')
    plt.ylabel('Number of samples')
    plt.title('Label distribution in dataset')
    plt.show()

# Example usage:
# Assuming 'trainloader' is your DataLoader for training data
#check_iid(trainloader)

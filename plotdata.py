import matplotlib.pyplot as plt


# Function to read data from a file
def read_data(file_path):
    epochs = []
    test_accuracies = []
    losses = []
    min_indices = []
    selected_indices = []

    with open(file_path, 'r') as f:
        # Skip the header
        next(f)

        for line in f:
            parts = line.strip().split('\t')
            epoch = int(parts[0])

            # Extract test accuracy and loss
            test_accuracy = float(parts[1])
            loss = parts[2]  # Read as string initially to handle 'NA'

            # Handle 'NA' in loss
            if loss == 'NA':
                loss = None
            else:
                loss = float(loss)

            min_index = parts[3]
            selected_index = parts[4]

            # Append data to lists
            epochs.append(epoch)
            test_accuracies.append(test_accuracy)
            losses.append(loss)

            if min_index != 'NA':
                min_indices.append(int(min_index))
            else:
                min_indices.append(None)

            if selected_index != 'NA':
                selected_indices.append(list(map(int, selected_index.split(','))))
            else:
                selected_indices.append(None)

    return epochs, test_accuracies, losses, min_indices, selected_indices

'''f'./result/new2_ldp_sgd_result_train_cifar_Krum_attack_xie_epsilon_0.1_addtime_50_malicious_11_epoch_50_users_0.25_sigma_{i}_C_3.txt'
    for i in range(11)
    f'./result/new2_ldp_sgd_result_train_cifar_Krum_attack_xie_epsilon_0.1_addtime_50_malicious_11_epoch_50_users_0.25_sigma_6_C_3.txt',
f'./result/Instancenorm_cifar_Krum_attack_xie_epsilon_0.1_addtime_50_malicious_11_epoch_50_users_0.25.txt'''
# File paths
file_paths = [

f'./result/gradresult_seed_cifar_Krum_attack_xie_epsilon_0.1_addtime_10_malicious_{i+1}_epoch_50_users_0.25.txt'
    for i in range(9,11)
]
file_paths.append('./result/gradresult_seed_cifar_Krum_attack_xie_epsilon_0.1_addtime_10_malicious_1_epoch_50_users_0.25.txt')


#r'cifar,Krum,malicious 1($\epsilon=0.1$)',
# Custom labels
custom_labels = [
rf'q = {i+1}'
    for i in range(9,11)


]
custom_labels.append(r'q = 1')


custom_loss_labels = [
rf'q = {i+1}'
    for i in range(9,11)

]
custom_loss_labels.append(r'q = 1')
# Initialize lists to store data from all files
all_epochs = []
all_test_accuracies = []
all_losses = []

# Read data from each file
for file_path in file_paths:
    epochs, test_accuracies, losses, _, _ = read_data(file_path)
    all_epochs.append(epochs)
    all_test_accuracies.append(test_accuracies)
    all_losses.append(losses)

# List of markers to use for different plots
markers = ['o', 'x', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D']

# Plot Test Accuracy from all files with different markers
plt.figure(figsize=(10, 6))
for i, test_accuracies in enumerate(all_test_accuracies):
    plt.plot(all_epochs[i], test_accuracies, marker=markers[i % len(markers)], label=custom_labels[i])

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('./result/zz_vs_f_Krum_test_accuracy.png')
plt.show()


# Plot Loss from all files
plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_losses):
    plt.plot(all_epochs[i], losses,marker=markers[i % len(markers)], label=custom_loss_labels[i])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs ')
plt.legend()
plt.grid(True)
plt.savefig('./result/zz_vs_f_Krum_train_loss.png')
plt.show()
##!/bin/bash

'''C=3  # Fixed sigma value

for sigma in {0..10}
do
    echo "Running with C=$C and sigma=$sigma"
    nice -n 19 python main_ldp_sgd_in_train.py --dataset cifar --num_channels 3 --model cnn --epoch 50 --gpu -1 --frac 0.25 --local_bs 10 --malicious 11 --Agg Krum --attack xie --epsilon 1 --addtime 10 --iid --sigma $sigma --C $C
done

#!/bin/bash

for sigma in {0..10}
do
    echo "Running with sigma=$sigma"
    nice -n 19 python main_ldp_sgd_in_train.py --dataset cifar --num_channels 3 --model cnn --epoch 50 --gpu -1 --frac 0.25 --local_bs 10 --malicious 11 --Agg Krum --attack xie --epsilon 0.1 --addtime 10 --iid --sigma $sigma --C 3
done
'''

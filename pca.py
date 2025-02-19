'''import numpy as np
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import OrderedDict

w_locals = torch.load('./save/w_locals1att.pth')
param_list = []

for client_params in w_locals:
    client_vector = []
    for param_tensor in client_params.values():
        client_vector.append(param_tensor.view(-1))  # Flatten each parameter tensor
    client_vector = torch.cat(client_vector).numpy()  # Convert to numpy array
    param_list.append(client_vector)

client_vectors = np.array(param_list)

pca = PCA(n_components=2)  # or n_components=3 for 3D plot
pca_result = pca.fit_transform(client_vectors)

# Plotting
plt.figure()
plt.scatter(pca_result[:, 0], pca_result[:, 1])
# Annotate each point with its index
for i, (x, y) in enumerate(pca_result):
    plt.text(x+0.02, y+0.02, str(i), fontsize=12, ha='right')
plt.title('PCA of 10 Client Model Parameters with 3 malicious model')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('./save/w_locals1att.png')'''
'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

# Load the client model parameters
#w_locals = torch.load('./save/w_locals_epoch_5_att_epsilon0_308.pth')
w_locals = torch.load('./save/w_locals_epoch_5_att_epsilon0.1_081.pth')
param_list = []
maliecious_indices = {0,1,8}

for i,client_params in enumerate(w_locals):
    client_vector = []
    for param_tensor in client_params.values():
        param_array = param_tensor.view(-1).numpy()
        if i in maliecious_indices:
            param_array *= 0
        client_vector.append(param_array)
    client_vector = np.concatenate(client_vector)  # Convert to a single numpy array
    param_list.append(client_vector)

client_vectors = np.array(param_list)

# Center the data
scaler = StandardScaler(with_std=False)  # Center the data, but don't scale it
client_vectors_centered = scaler.fit_transform(client_vectors)

# Perform PCA
pca = PCA(n_components=2)  # or n_components=3 for 3D plot
pca_result = pca.fit_transform(client_vectors_centered)

# Find the closest pair of points
min_distance = float('inf')
closest_pair = (0, 0)

for i in range(len(pca_result)):
    for j in range(i + 1, len(pca_result)):
        distance = np.linalg.norm(pca_result[i] - pca_result[j])
        if distance < min_distance:
            min_distance = distance
            closest_pair = (i, j)



# Plotting
plt.figure(figsize=(10, 8))

# Plot all points
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='blue', alpha=0.5)

# Highlight the closest pair
for i, (x, y) in enumerate(pca_result):
    color = 'red' if i in closest_pair else 'blue'
    plt.scatter(x, y, s=100, color=color)
    plt.text(x + 0.01, y + 0.01, str(i), fontsize=12, ha='left', alpha=0.7)

plt.title('PCA of 10 Client Model Parameters with 3 Malicious Models')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
#plt.savefig('./save/w_locals_epoch_5_att_epsilon0_308.png')
plt.show()'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

# Load the client model parameters
#w_locals = torch.load('./save/w_locals_epoch_5_att_epsilon0_308.pth')
#w_locals = torch.load('./save/w_locals_epoch_5_att_epsilon1_378.pth')
w_locals = torch.load('save/Instancenorm_sgdgrad_locals_addtime_4_epoch_5_att_epsilon0.1_823.pth')
param_list = []
original_indices = []
# List of indices of malicious clients to scale
malicious_indices = {3,2,8}
epsilon = 0.1

for i, client_params in enumerate(w_locals):
    client_vector = []
    for param_tensor in client_params.values():
        param_array = param_tensor.view(-1).numpy()  # Flatten each parameter tensor and convert to numpy array
        if i in malicious_indices:
             param_array = epsilon* param_array # Scale down the parameters of malicious clients
        client_vector.append(param_array)
    client_vector = np.concatenate(client_vector)  # Convert to a single numpy array
    param_list.append(client_vector)
    original_indices.append(i)
client_vectors = np.array(param_list)

# Perform PCA
pca = PCA(n_components=2)  # or n_components=3 for 3D plot
pca_result = pca.fit_transform(client_vectors)
honest_result = [item for index, item in enumerate(pca_result) if index not in malicious_indices]
malicious_result = [item for index, item in enumerate(pca_result) if index in malicious_indices]
# Compute the mean of the points
mean_point = np.mean(honest_result, axis=0)

# Compute the mean multiplied by -0.1
mean_point_neg_scaled = mean_point * -epsilon

# Plotting
plt.figure(figsize=(10, 8))

# Plot all points
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='blue', alpha=0.5)
# Highlight the closest pair
for i, (x, y) in enumerate(pca_result):
    #color = 'red' if i in closest_pair else 'blue'
    #plt.scatter(x, y, s=100, color=color)
    plt.text(x + 0.01, y + 0.01, str(original_indices[i]), fontsize=12, ha='left', alpha=0.7)
# Plot the mean point
plt.scatter(mean_point[0], mean_point[1], color='black', s=200, label='Mean Point of honest')

# Plot the mean point multiplied by -0.1
#plt.scatter(mean_point_neg_scaled[0], mean_point_neg_scaled[1], color='green', s=200, label='Mean Point * -{}'.format(epsilon))


plt.title('PCA of gradients with epsilon {}_useInstancenorm'.format(epsilon))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend()
plt.savefig('./save/Instancenorm_grad_locals_epoch_5_att_epsilon{}_823.png'.format(epsilon))
plt.show()







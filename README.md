# Federated Learning for Privacy-Preserving Machine Learning

In recent years, artificial intelligence has garnered increasing attention. At the core of AI lies machine learning, which has been widely applied in areas such as finance, healthcare, autonomous driving, online shopping, and job searching. While these advances have brought significant convenience, they also raise pressing privacy concerns. Traditional centralized machine learning requires aggregating users’ data on a server—data that often includes sensitive personal details (e.g., demographic information, shopping habits, and even medical records). A breach during this collection process could threaten both personal and financial security.

To address these issues, Google introduced Federated Learning in 2017. Federated learning is a decentralized training framework that enables multiple data holders (e.g., smartphones, IoT devices) to collaboratively train a model without sharing raw data. Instead, only intermediate parameters are exchanged. In an ideal scenario, this decentralized model achieves performance comparable to a model trained on centrally aggregated data.

However, federated learning is not without its challenges. Although it mitigates the risks associated with centralizing private data, the exchange of intermediate parameters can still expose sensitive information. For instance, research has demonstrated that original data can sometimes be partially reconstructed from gradients or that one can infer the presence of a specific participant’s data from these shared parameters.

Furthermore, the framework is vulnerable to malicious participants. With only model parameters being exchanged, attackers may conduct model poisoning attacks by injecting manipulated parameters. In one scenario, a participant could alter the model weight \( w_1 \) to disproportionately influence the global model \( w' \).

In this repository, we present a comprehensive approach that:
- **Analyzes the aggregation technique FedAvg.**
- **Introduces a robust aggregation method, KRUM,** to defend against Byzantine attacks.
- **Incorporates differential privacy** to further safeguard user data during the training process.

---

## Repository Structure

- **README.md**  
  This file, providing an overview of the project and instructions.
  
- **src/**  
  Source code for the federated learning framework, including implementations of:
  - *FedAvg* for standard federated aggregation.
  - *KRUM* for robust aggregation against Byzantine failures.
  - Differential privacy modules.

- **configs/**  
  Configuration files (in JSON format) that define experimental parameters and hyperparameters.

- **experiments/**  
  Scripts and notebooks to reproduce experiments and evaluate model performance.

- **data/**  
  Sample datasets to simulate federated learning scenarios.

- **docs/**  
  Additional documentation and notes on methodology and experimental results.

---

## Installation

### Requirements
- **Python 3.x**
- Required libraries: TensorFlow or PyTorch (choose one based on your implementation), NumPy, SciPy, etc.

Install dependencies using:
```bash
pip install -r requirements.txt


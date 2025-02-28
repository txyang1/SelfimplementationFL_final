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
'''bash
for malicious in {1..11}; do     echo "Running with malicious=$malicious";     nice -n 19 python main_sgd.py --dataset cifar --num_channels 3 --model cnn --epoch 50 --gpu -1 --frac 0.25 --local_bs 10 --malicious $malicious --Agg Krum --attack xie --epsilon 1 --addtime 
10 --iid ; done

### Requirements
- **Python 3.x**
- Required libraries: TensorFlow or PyTorch (choose one based on your implementation), NumPy, SciPy, etc.
- 

References
bibtex
Copy
@misc{imag,
  author = {securemymind},
  title = {privacy image},
  year = {2021},
  url = {https://blog.securemymind.com/how-to-protect-privacy.html},
  note = {Accessed: 2024-10-20}
}

@Article{fi13030073,
  AUTHOR = {Zhou, Xingchen and Xu, Ming and Wu, Yiming and Zheng, Ning},
  TITLE = {Deep Model Poisoning Attack on Federated Learning},
  JOURNAL = {Future Internet},
  VOLUME = {13},
  YEAR = {2021},
  NUMBER = {3},
  ARTICLE-NUMBER = {73},
  URL = {https://www.mdpi.com/1999-5903/13/3/73},
  ISSN = {1999-5903},
  DOI = {10.3390/fi13030073}
}

@article{chen2016revisiting,
  title={Revisiting distributed synchronous SGD},
  author={Chen, Jianmin and Pan, Xinghao and Monga, Rajat and Bengio, Samy and Jozefowicz, Rafal},
  journal={arXiv preprint arXiv:1604.00981},
  year={2016}
}

@Inbook{Yu2023,
  author = {Yu, Shui and Cui, Lei},
  title = {Poisoning Attacks and Counterattacks in Federated Learning},
  bookTitle = {Security and Privacy in Federated Learning},
  year = {2023},
  publisher = {Springer Nature Singapore},
  address = {Singapore},
  pages = {37--54},
  isbn = {978-981-19-8692-5},
  doi = {10.1007/978-981-19-8692-5_3},
  url = {https://doi.org/10.1007/978-981-19-8692-5_3}
}

@inproceedings{narayanan2008robust,
  title = {Robust de-anonymization of large sparse datasets},
  author = {Narayanan, Arvind and Shmatikov, Vitaly},
  booktitle = {2008 IEEE Symposium on Security and Privacy (sp 2008)},
  pages = {111--125},
  year = {2008},
  organization = {IEEE}
}

@inproceedings{melis2019exploiting,
  title = {Exploiting unintended feature leakage in collaborative learning},
  author = {Melis, Luca and Song, Congzheng and De Cristofaro, Emiliano and Shmatikov, Vitaly},
  booktitle = {2019 IEEE symposium on security and privacy (SP)},
  pages = {691--706},
  year = {2019},
  organization = {IEEE}
}
Copy


Install dependencies using:
```bash
pip install -r requirements.txt


import csv

def csv_to_txt(csv_file_path, txt_file_path):
    with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        with open(txt_file_path, 'w', newline='') as txt_file:
            for row in csv_reader:
                txt_file.write('\t'.join(row) + '\n')

# Example usage
csv_file_path = './result/new_Krum_attack_xie_epsilon_0.5_addtime_10_malicious_11_epoch_20_users_0.25.csv'
txt_file_path = './result/mnist_Krum_attack_xie_epsilon_0.5_addtime_10_malicious_11_epoch_20_users_0.25.txt'
csv_to_txt(csv_file_path, txt_file_path)

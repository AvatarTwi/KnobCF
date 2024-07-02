import pickle


def read_attr_val_dict(file_path):
    with open(file_path, 'rb') as f:
        lines = pickle.load(f)
    print(lines)

if __name__ == '__main__':
    file_path = 'tpch/attr_val_dict.pickle'
    attr_val_dict = read_attr_val_dict(file_path)
    print(attr_val_dict)
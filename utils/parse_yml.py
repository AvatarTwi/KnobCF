import yaml


def read_yaml(path):
    with open(path, 'r') as file:
        data = file.read()
        # result = yaml.load(data)
        result = yaml.load(data, Loader=yaml.FullLoader)
        return result


def read_yaml_k(conf_path, k):
    # 打开文件
    with open(conf_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        try:
            # 判断传入的n是否在存在
            if k in data.keys():
                return data[k]
            else:
                print(f"n：{k}不存在")
        except Exception as e:
            print(f"key值{e}不存在")

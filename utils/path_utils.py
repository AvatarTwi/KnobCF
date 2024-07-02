import os
import shutil
import time


class PathUtils:
    @staticmethod
    def path_build(root):
        if not os.path.exists(root):
            os.makedirs(root)

    @staticmethod
    def path_del(root):
        if os.path.exists(root):
            shutil.rmtree(root)

    @staticmethod
    def list_files_in_directory(directory):
        try:
            # 获取目录下的所有文件和子目录
            files = os.listdir(directory)

            return [directory + file for file in files]
        except Exception as e:
            print(f"发生错误：{e}")

    @staticmethod
    def return_files_in_directory(directory):
        try:
            # 获取目录下的所有文件和子目录
            files = os.listdir(directory)

            return files
        except Exception as e:
            print(f"发生错误：{e}")

    @staticmethod
    def concatenate_files_in_folder(folder_path, output_file):
        with open(output_file, 'w') as output:
            for idx, filename in enumerate(os.listdir(folder_path)):
                if idx == 3:
                    break
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as input_file:
                        output.write(input_file.read())


if __name__ == '__main__':
    # 替换folder_path为你实际的文件夹路径，output_file为输出文件路径
    folder_path = '../vm/data/tpcc4base'
    output_file = '../vm/data/tpcc4base/serverlog0.txt'

    PathUtils.concatenate_files_in_folder(folder_path, output_file)

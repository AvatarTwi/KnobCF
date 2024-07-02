import os
import stat
import time
from stat import S_ISDIR

import paramiko

from utils.parse_yml import read_yaml


def get_all_files_in_local_dir(local_dir):
    """递归获取当前目录下所有文件目录"""
    all_files = []
    # 获取当前指定目录下的所有目录及文件，包含属性值
    files = os.listdir(local_dir)
    for x in files:
        # local_dir目录中每一个文件或目录的完整路径
        filename = os.path.join(local_dir, x)
        # 如果是目录，则递归处理该目录
        if os.path.isdir(filename):
            all_files.extend(get_all_files_in_local_dir(filename))
        else:
            all_files.append(filename)
    return all_files


# ------获取远端linux主机上指定目录及其子目录下的所有文件------
def get_all_files_in_remote_dir(sftp, remote_dir):
    # 保存所有文件的列表
    all_files = list()

    # 去掉路径字符串最后的字符'/'，如果有的话
    if remote_dir[-1] == '/':
        remote_dir = remote_dir[0:-1]

    # 获取当前指定目录下的所有目录及文件，包含属性值
    files = sftp.listdir_attr(remote_dir)
    print(remote_dir)
    print(files)
    for x in files:
        # remote_dir目录中每一个文件或目录的完整路径
        filename = remote_dir + '/' + x.filename
        # 如果是目录，则递归处理该目录，这里用到了stat库中的S_ISDIR方法，与linux中的宏的名字完全一致
        if S_ISDIR(x.st_mode):
            all_files.extend(get_all_files_in_remote_dir(sftp, filename))
        else:
            all_files.append(filename)
    return all_files


class LinuxConnector:

    def __init__(self, vm_conf='conf/vm_conf.yaml'):
        vm_config = read_yaml(vm_conf)
        vm = vm_config['server']
        self.ip = vm['host']
        self.port = vm['port']
        self.root_name = vm['root_name']
        self.root_passwd = vm['root_password']
        self.result = []

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ssh.connect(hostname=self.ip, port=self.port,
                         username=self.root_name, password=self.root_passwd,
                         allow_agent=False, look_for_keys=False)

        self.tranport = paramiko.Transport((self.ip, self.port))

        self.tranport.connect(username=self.root_name, password=self.root_passwd)
        self.sftp = paramiko.SFTPClient.from_transport(self.tranport)

    def get_ssh(self):
        return self.ssh

    def close(self):
        self.ssh.close()
        self.sftp.close()

    def exec_command(self, command):
        _, stdout, stderr = self.ssh.exec_command(command)
        result = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')

        return result, err

    # 执行多条命令,注意传入的参数有个list
    def exec_multi_cmd(self, cmds):
        cmd = " ; ".join(cmds)
        print(cmd)
        _, stdout, stderr = self.ssh.exec_command(cmd, get_pty=True)
        result = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')

        return result, err

    def upload_file_path(self, remote_path, localfile_path):
        # 待上传目录名
        local_pathname = os.path.split(localfile_path)[-1]
        # 上传远程后的目录名
        real_remote_Path = remote_path + local_pathname

        # 判断是否存在，不存在则创建
        try:
            self.sftp.stat(remote_path)
        except Exception as e:
            self.ssh.exec_command("mkdir -p %s" % remote_path)

        self.ssh.exec_command("mkdir -p %s" % real_remote_Path)

        toremote_path = remote_path + local_pathname

        self.sftp.put(localfile_path, toremote_path)

    """
    :param local_path:待上传文件夹路径
    :param remote_path:远程路径
    :return:
    """
    def upload_dir_path(self, remote_path, local_path):
        # 待上传目录名
        local_pathname = os.path.split(local_path)[-1]
        # 上传远程后的目录名
        real_remote_Path = remote_path + local_pathname

        # 判断是否存在，不存在则创建
        try:
            self.sftp.stat(remote_path)
        except Exception as e:
            self.ssh.exec_command("mkdir -p %s" % remote_path)

        self.ssh.exec_command("mkdir -p %s" % real_remote_Path)
        # 获取本地文件夹下所有文件路径
        all_files = get_all_files_in_local_dir(local_path)
        # 依次判断远程路径是否存在，不存在则创建，然后上传文件
        for file_path in all_files:
            # 统一win和linux 路径分隔符
            file_path = file_path.replace("\\", "/")
            # 用本地根文件夹名分隔本地文件路径，取得相对的文件路径
            off_path_name = file_path.split(local_pathname)[-1]
            # 取得本地存在的嵌套文件夹层级
            abs_path = os.path.split(off_path_name)[0]
            # 生产期望的远程文件夹路径
            reward_remote_path = real_remote_Path + abs_path
            # 判断期望的远程目录是否存在，不存在则创建
            try:
                self.sftp.stat(reward_remote_path)
            except Exception as e:
                self.ssh.exec_command("mkdir -p %s" % reward_remote_path)
            # 待上传的文件名
            abs_file = os.path.split(file_path)[1]
            # 上传后的远端路径，文件名不变
            to_remote = reward_remote_path + '/' + abs_file
            time.sleep(0.1)
            self.sftp.put(file_path, to_remote)
            print(file_path, to_remote)

    def download_file_path(self, remote_file, local_file):
        self.sftp.get(remote_file, local_file)

        return self.result

    def download_dir_path(self, remote_path, local_path):
        if os.path.isdir(local_path):
            pass
        else:
            os.makedirs(local_path)

        # check file/dir
        isfile_list = []
        isdir_list = []
        remote_allfiles = self.sftp.listdir_attr(remote_path)

        for f in remote_allfiles:
            remote_file = remote_path + "/" + f.filename
            dirname = remote_path.split("/")[-1]
            local_file = local_path + "/" + dirname + "/" + f.filename
            if stat.S_ISDIR(f.st_mode):
                isdir_list.append(remote_file)
            else:
                isfile_list.append(remote_file)

        # first step: get file
        for file_name in isfile_list:
            remote_file = file_name
            try:
                os.makedirs(os.path.join(local_path, file_name.split("/")[-2]))
            except:
                pass

            local_file = local_path + "/" + file_name.split("/")[-2] + "/" + file_name.split("/")[-1]
            self.sftp.get(remote_file, local_file)

        # second step: loop dir
        n = 0

        for dir_name in isdir_list:
            if n > 0:
                local_path = local_path
            else:
                local_path = local_path + "/" + dir_name.split("/")[-2]
            remote_path = dir_name
            self.download_dir_path(remote_path, local_path)
            n += 1

        return self.result

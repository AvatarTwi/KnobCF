from time import sleep

from manager.linux_manager import LinuxConnector
from utils.parse_yml import read_yaml


class PgManager:
    def __init__(self, args):
        self.lc = LinuxConnector(args.vm_conf)
        self.db = read_yaml(args.db_conf)['database']

    def start_db(self):
        self.lc.exec_command('rm -rf ' + self.db['datadir'] + 'serverlog')
        self.lc.exec_command('service postgresql start')
        sleep(2)

    def stop_db(self):
        self.lc.exec_command('service postgresql stop')
        sleep(2)

    def restart_db(self):
        self.stop_db()
        self.start_db()

    def update_db_config(self, knob_dict):
        self.stop_db()
        default_conf = self.db['local-conf'][0]
        pgsql_conf = self.db['local-conf'][1]
        lines = open(default_conf, 'r').readlines()

        for key, value in knob_dict.items():
            lines.append(key + ' = ' + str(value) + '\n')
        postgresql_conf = ''.join(lines)

        auto_explain = ["shared_preload_libraries = 'auto_explain'",
                        "auto_explain.log_min_duration = 0",
                        "auto_explain.log_format = json",
                        "auto_explain.log_analyze = true",
                        "auto_explain.log_timing = true",
                        "auto_explain.log_verbose = true"]
# shared_preload_libraries = 'auto_explain'
# auto_explain.log_min_duration = 0
# auto_explain.log_format = json
# auto_explain.log_analyze = true
# auto_explain.log_timing = true
# auto_explain.log_verbose = true

# log_statement = 'all'
# logging_collector = on
# log_directory = 'log'
# log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
# log_file_mode = 0600
        log = [
            "log_statement = 'all'",
            "logging_collector = on",
            "log_directory = 'log'",
            "log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'",
            "log_file_mode = 0600"
        ]
        postgresql_conf += '\n'.join(auto_explain)

        with open(pgsql_conf, 'w') as f:
            f.write(postgresql_conf)

        self.lc.upload_file_path(self.db['datadir'], pgsql_conf)
        self.start_db()
        self.lc.exec_command("echo 1 > /proc/sys/vm/drop_caches;")

    def close(self):
        self.lc.close()

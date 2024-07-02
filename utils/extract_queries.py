import re


def extract_queries(name):
    queries = []
    with open('../vm/postgresql-2024-06-13_084412.log', 'r') as f:
        lines = f.readlines()
        for l in range(len(lines)):
            if 'LOG:  execute' in lines[l] and \
                    ('INSERT' in lines[l] or 'UPDATE' in lines[l] or 'DELETE' in lines[l] or 'SELECT' in lines[l]):
                query = lines[l].split(': ')[-1].replace('\n', '')
                parameters = lines[l + 1].split('parameters: ')[1].replace('\n', '').split('\', $')
                for i in range(len(parameters)):
                    if i != len(parameters) - 1:
                        parameters[i] = parameters[i]+'\''
                    if i != 0:
                        parameters[i] = '$' + parameters[i]
                parameters_list = {re.compile(r'\$\d').findall(p)[0]:
                                       p.replace(re.compile(r"\$\d\s=\s").findall(p)[0], '')
                                   for p in parameters}
                for key, value in parameters_list.items():
                    query = query.replace(key, value)
                queries.append(query)

    with open('../vm/'+name+'.sql', 'w') as f:
        f.write(';\n'.join(queries))



def remove_DEL_UPD_INS(file_path):
    res = []
    with open('../vm/' + file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'DELETE' in line or 'UPDATE' in line or 'INSERT' in line:
                continue
            else:
                res.append(line)
    with open('../vm/' + file_path, 'w') as f:
        f.write(''.join(res))

if __name__ == '__main__':

    extract_queries('ycsb_c')
    # remove_DEL_UPD_INS('tpcc.sql')

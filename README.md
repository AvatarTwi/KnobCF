# KnobCF: Uncertainty-aware Knob Tuning

KnobCF is an uncertainty-based knob tuning method that improves the time-precision efficiency of the knob tuning process. 
This is the implementation described in the paper: 

KnobCF: Uncertainty-aware Knob Tuning.[PDF](https://arxiv.org/pdf/2407.02803)

![tuning_gj](https://typora-picpool-1314405309.cos.ap-nanjing.myqcloud.com/img/tuning_gj.png)


## Source Code Structure

- `benchmarks/`: the queries in benchmark for the experiments
- `conf/`
  - `iniconf/`: the configuration file for the experiments
  - `bm_conf.yaml`: the configuration file for the benchmark
  - `db_conf.yaml`: the configuration file for the database
  - `vm_conf.yaml`: the configuration file for the virtual machine
- `dataset/*`: the attribute template for different workloads
- `executors/`: the executor for the ddpg
- `featurization/`: 
  - `benchmark_tools/`: the tools for the benchmark
  - `BinaryNet/`: the binary network for zero-shot
  - `datasets/`: the attribute and stats for different workloads
  - `zero_shot_featurization/`: the featurization for the zero-shot, [ZeroShot](https://github.com/DataManagementLab/zero-shot-cost-estimation)
- `manager/`: the manager for database connection, and virtual machine connection
- `model/`: 
  - `corr/`: 
    - `Configurations/`: define the class of configuration
    - `ConfigurationPool/`: define the class of ConfigurationPool
    - `Interface/`: define the class of Interface, connect the KnobEstimator and the KnobTuner
    - `knob_estimator/`: define the class of KnobEstimator
    - ..
  - `query_encoder_model/`: the models for the query encoder
    - `QueryFormerModule/`: the module for the KnobCF(QueryFormer), [QueryFormer](https://github.com/zhaoyue-ntu/QueryFormer)
    - `KnobCFModule/`: the query embedding for KnobCF
- `optimizer_model/`: contains the model for the optimizer, such as [ddpg, smac](https://github.com/uw-mad-dash/llamatune)
- `reward/`: get reward from the database
- `space/`: the configuration space for the database
- `utils/`: the utility functions
- `main.py`: the main script used for running experiments
- `run_optimizer.py`: the main script used for running the optimizer

## Environment

Tested with python 3.10. To install the required packages, please run the following command:

```shell
conda install --yes --file requirements.txt
```

## Run Experiments

### Data Preparation

1. TPCH
   1. Download the TPCH benchmark from [TPCH](https://www.tpc.org/tpch/)
   2. Create a database and load the TPCH benchmark, set the scale factor to 10
2. TPCC
   1. Download the TPCC benchmark from [TPCC](https://www.tpc.org/tpcc/)
   2. set warehouses=100 and loadWorkers=4, create a database and load the TPCC benchmark
3. YCSB
   1. Download the YCSB benchmark from [YCSB](https://github.com/brianfrankcooper/YCSB)
   2. Set ‘recordcount= 1000000‘, ‘operationcount = 100, create a database and load the YCSB benchmark
      1. bin/ycsb.bat load basic -P workloads\workloada 
      2. bin/ycsb.bat run basic -P workloads\workloada 
   3. Set ‘recordcount= 1000000‘, ‘operationcount = 100, create a database and load the YCSB-b benchmark
      1. bin/ycsb.bat load basic -P workloads\workloadb 
      2. bin/ycsb.bat run basic -P workloads\workloadb
5. JOB-light
   1. Download the JOB-light benchmark from [imdb.tgz]([http://homepages.cwi.nl/~boncz/job/imdb.tgz](https://link.zhihu.com/?target=http%3A//homepages.cwi.nl/~boncz/job/imdb.tgz))
   2. Create a database and load the JOB-light benchmark

### Main Script

The main script used for running experiments is `main.py`. 

With setting some parameters, one can run the experiments with different models and workloads. The parameters are as follows:

1. `mode`: there types of mode:
    - run db: run_db, run_db_default
    - train query encoder: train_query_encoder, eval_query_encoder
    - train knob estimator: train_knob_estimator, eval_knob_estimator, trans_knob_estimator
2. `workload`: 
    - `TPCH`: the TPCH workload
    - `TPCC`: the TPCC workload
    - `ycsb`: the ycsb workload
    - `ycsb-b`: the ycsb-b workload
    - `job-light`: the job-light workload
3. `num_clusters`: set the output dimension of knob estimator
4. `optimizer_type`: set the optimizer type, such as `ddpg`, `smac`
5. `query_encoder_model`: set the query encoder model, such as KnobCF(few-shot), KnobCF, query_former

## Usage Examples

To collect the training data of TPCC, one should set the following parameters:
```
mode: run_db
workload: TPCC
```
then run main.py

elif one could set the following parameters to get the distribution of default configuration in TPCC:
```
mode: run_db_default
workload: TPCC
```
then run main.py

To train the query encoder KnobCF of TPCC, one should set the following parameters:
```
mode: train_query_encoder
workload: TPCC
query_encoder_model: KnobCF
```
then run main.py

elif one could set the following parameters to train the query encoder QueryFormer of TPCC:
```
mode: train_query_encoder
workload: TPCC
query_encoder_model: QueryFormer
```
then run main.py

To train the knob estimator of TPCC, with output dimension 16, one should set the following parameters:
```
mode: train_knob_estimator
workload: TPCC
query_encoder_model: KnobCF
num_clusters: 16
```
then run main.py

To train the optimizer ddpg of TPCC with KnobCF as knob estimator, one should set the following parameters:
```
mode: run_db
workload: TPCC
optimizer_type: ddpg
query_encoder_model: KnobCF
num_clusters: 16
```
then run main.py

## Reference

If you find this repository useful in your work, please cite our paper:

```
https://arxiv.org/abs/2407.02803
```

## Code Citations

We utilize some open source libraries to implement our experiments. The specific citation is as follows:

```
Github repository: SHAP. https://github.com/shap/shap.
Github repository: ZeroShot. https://github.com/DataManagementLab/zero-shot-cost-estimation
Github repository: QueryFormer. https://github.com/zhaoyue-ntu/QueryFormer
Github repository: ddpg,smac. https://github.com/uw-mad-dash/llamatune
```




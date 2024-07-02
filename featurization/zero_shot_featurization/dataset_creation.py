from featurization.zero_shot_featurization.plan_graph_batching.postgres_plan_batching import postgres_plan_collator
from featurization.zero_shot_featurization.utils.feature_statistics import gather_feature_statistics

def dataset_creation(parsed_runs, plan_featurization_name='PostgresTrueCardDetail'):

    # split plans into train/test/validation
    train_dataset, database_statistics = parsed_runs['parsed_plans'], parsed_runs['database_stats']

    # postgres_plan_collator does the heavy lifting of creating the graphs and extracting the features and thus requires both
    # database statistics but also feature statistics
    feature_statistics = gather_feature_statistics(parsed_runs)

    graph, features, labels, sample_idxs = postgres_plan_collator(train_dataset,
                                                                  feature_statistics=feature_statistics,
                                                                  db_statistics=database_statistics,
                                                                  plan_featurization_name=plan_featurization_name)



    return (graph, features), feature_statistics

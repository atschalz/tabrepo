from autogluon.core.models import DummyModel

from tabrepo.benchmark.experiment import AGModelBagExperiment, YamlExperimentSerializer
from tabrepo.utils.config_utils import ConfigGenerator

def get_config_generator(model_name):
    if model_name == 'LightGBM':
        from tabrepo.models.lightgbm.generate import gen_lightgbm as config_generator
    elif model_name == 'TabICL':
        from tabrepo.models.tabicl.generate import gen_tabicl as config_generator
    elif model_name == 'TabDPT':    
        from tabrepo.models.tabdpt.generate import gen_tabdpt as config_generator
    elif model_name == 'KNN':   
        from tabrepo.models.knn.generate import gen_knn as config_generator
    elif model_name == 'FastAI':
        from tabrepo.models.fastai.generate import gen_fastai as config_generator
    elif model_name == 'ExtraTrees':
        from tabrepo.models.extra_trees.generate import gen_extratrees as config_generator
    elif model_name == 'CatBoost':
        from tabrepo.models.catboost.generate import gen_catboost as config_generator
    elif model_name == 'XGBoost':
        from tabrepo.models.xgboost.generate import gen_xgboost as config_generator
    elif model_name == 'NN_Torch':
        from tabrepo.models.nn_torch.generate import gen_nn_torch as config_generator
    elif model_name == 'RandomForest':
        from tabrepo.models.random_forest.generate import gen_randomforest as config_generator
    elif model_name == 'Linear':
        from tabrepo.models.lr.generate import gen_linear as config_generator
    elif model_name == 'TabPFNv2':
        from tabrepo.models.tabpfnv2.generate import gen_tabpfnv2 as config_generator
    elif model_name == 'RealMLP':
        from tabrepo.models.realmlp.generate import gen_realmlp as config_generator
    elif model_name == 'EBM':
        from tabrepo.models.ebm.generate import gen_ebm as config_generator
    elif model_name == 'TabM':
        from tabrepo.models.tabm.generate import gen_tabm as config_generator
    elif model_name == 'ModernNCA':
        from tabrepo.models.modernnca.generate import gen_modernnca as config_generator
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return config_generator 


if __name__ == '__main__':
    use_models = ['CatBoost']
    n_random_configs = 200
    n_random_configs_baselines = 50
    include_dummy = False
    preprocessor_name = 'all_in_one'
    config_suffix = 'all_in_one'
    time_limit = 7200

    # Dummy (constant predictor)
    if include_dummy:
        experiments_dummy = ConfigGenerator(model_cls=DummyModel, search_space={}, manual_configs=[{}]).generate_all_bag_experiments(num_random_configs=0, time_limit=time_limit)
        experiments_lst = [experiments_dummy]
    else:
        experiments_lst = []

    for model_name in use_models:
        config_generator = get_config_generator(model_name=model_name)
        if model_name in ['LinearModel', 'KNN']:  # TODO: Check whether these models are GPU only
            experiments = config_generator.generate_all_bag_experiments(num_random_configs=n_random_configs_baselines+1, reuse_tabarena=True, preprocessor_name=preprocessor_name)
        elif model_name in ['TabICL', 'TabDPT']:  # TODO: Check whether these models are GPU only
            experiments = config_generator.generate_all_bag_experiments(num_random_configs=0, reuse_tabarena=True, preprocessor_name=preprocessor_name)
        else:
            experiments = config_generator.generate_all_bag_experiments(num_random_configs=n_random_configs+1, reuse_tabarena=True, preprocessor_name=preprocessor_name)
        experiments_lst.append(experiments)

    experiments_all: list[AGModelBagExperiment] = [exp for exp_family_lst in experiments_lst for exp in exp_family_lst]

    # Verify no duplicate names
    experiment_names = set()
    for experiment in experiments_all:
        if experiment.name not in experiment_names:
            experiment_names.add(experiment.name)
        else:
            raise AssertionError(f"Found multiple instances of experiment named {experiment.name}. All experiment names must be unique!")

    YamlExperimentSerializer.to_yaml(experiments=experiments_all, path=f"configs_all_{config_suffix}.yaml")

    from tabrepo.models.automl import generate_autogluon_experiments
    experiments_autogluon = generate_autogluon_experiments()

    YamlExperimentSerializer.to_yaml(experiments=experiments_autogluon, path=f"configs_autogluon_{config_suffix}.yaml")

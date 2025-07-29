import os
import sys
sys.path.append("/home/ubuntu/cat_detection/tabrepo/tabflow")

print(os.getcwd())
print(sys.path)
from tabflow.cli.launch_jobs import JobManager
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
# from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
"""
bash ./tabrepo/tabflow/docker/build_docker.sh tabarena tabarena-neerick 763104351884 097403188315 us-west-2
 
aws s3 cp --recursive "s3://prateek-ag/neerick-exp-3/" ../data/neerick-exp-3/ --exclude "*.log"
https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs 
"""
# !!!!! CHANGED TO test-ftd (priorly test-csv) !!!!!
docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/andrej-test:test-ftd"
# docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/pmdesai:mlflow-tabrepo"
sagemaker_role = "arn:aws:iam::097403188315:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145"

##########################
# def get_benchmark_metadata(benchmark: str, subset: list[str]=None) -> pd.DataFrame:
#     if benchmark == "TabZilla":
#         from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
#         tids, dids = get_benchmark_dataIDs("TabZilla")
#         metadata = get_metadata_df(tids, dids)    
#     elif benchmark == "Grinsztajn":
#         from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
#         tids, dids = get_benchmark_dataIDs("Grinsztajn")
#         metadata = get_metadata_df(tids, dids)    
#     elif benchmark == "TabArena":
#         metadata = load_task_metadata()        
#     else:
#         raise ValueError(f"Unknown benchmark: {benchmark}")

#     if subset is not None:
#         metadata = metadata[metadata["name"].isin(subset)]

#     return metadata

# def get_model(model_name: str) -> AbstractModel:
#     return tabrepo_model_register.key_to_cls(model_name)  
##########################

batch_sizes = {'airfoil_self_noise': 64,
 'Amazon_employee_access': 16,
 'anneal': 64,
 'Another-Dataset-on-used-Fiat-500': 128,
 'APSFailure': 16,
 'bank-marketing': 32,
 'Bank_Customer_Churn': 32,
 'Bioresponse': 16,
 'blood-transfusion-service-center': 128,
 'churn': 64,
 'coil2000_insurance_policies': 32,
 'concrete_compressive_strength': 128,
 'credit-g': 128,
 'credit_card_clients_default': 32,
 'customer_satisfaction_in_airline': 32,
 'diabetes': 128,
 'Diabetes130US': 32,
 'diamonds': 32,
 'E-CommereShippingData': 32,
 'Fitness_Club': 128,
 'Food_Delivery_Time': 32,
 'GiveMeSomeCredit': 64,
 'hazelnut-spread-contaminant-detection': 64,
 'healthcare_insurance_expenses': 128,
 'heloc': 64,
 'hiva_agnostic': 32,
 'houses': 64,
 'HR_Analytics_Job_Change_of_Data_Scientists': 32,
 'in_vehicle_coupon_recommendation': 32,
 'Is-this-a-good-customer': 64,
 'kddcup09_appetency': 16,
 'Marketing_Campaign': 64,
 'maternal_health_risk': 128,
 'miami_housing': 32,
 'NATICUSdroid': 32,
 'online_shoppers_intention': 32,
 'physiochemical_protein': 32,
 'polish_companies_bankruptcy': 32,
 'qsar-biodeg': 64,
 'QSAR-TID-11': 64,
 'QSAR_fish_toxicity': 64,
 'SDSS17': 32,
 'seismic-bumps': 64,
 'splice': 64,
 'students_dropout_and_academic_success': 64,
 'superconductivity': 16,
 'taiwanese_bankruptcy_prediction': 32,
 'website_phishing': 128,
 'wine_quality': 128,
 'MIC': 64,
 'jm1': 64}

if __name__ == "__main__":
    task_metadata = load_task_metadata()

    experiment_name = "test"
    max_concurrent_jobs = 2000
    batch_size = 8
    wait = True
    s3_bucket = "prateek-ag"
    region_name = "us-west-2"
    instance_type = "ml.m6i.2xlarge" #"ml.g6.2xlarge" 

    methods_file = "configs_all_new.yaml"  # TODO: Need to create this file
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)

    datasets = list(task_metadata["name"])

    # raise AssertionError("No datasets found")

    # toy run
    datasets = datasets[:1]
    folds = list(range(3))
    repeats = list(range(1))
    methods = methods[:1]

    print(datasets)
    print()

    job_manager = JobManager(
        experiment_name=experiment_name,
        task_metadata=task_metadata,
        methods_file=methods_file,
        max_concurrent_jobs=max_concurrent_jobs,
        s3_bucket=s3_bucket,
        wait=wait,
        instance_type=instance_type,
        batch_size=batch_size,
        sagemaker_role=sagemaker_role,
        docker_image_uri=docker_image_uri,
    )
    
    tasks_batch_combined = []
    for dataset_name in datasets:

        tasks_dense = job_manager.get_tasks_dense(
            datasets=[dataset_name],
            repeats=repeats,
            folds=folds,
            methods=methods,
        )
        # uncached_tasks = job_manager.filter_to_only_uncached_tasks(tasks=tasks_dense, verbose=True)

        tasks_batch = job_manager.batch_tasks(tasks=tasks_dense, batch_size=batch_sizes[dataset_name])
        tasks_batch_combined.extend(tasks_batch)

    uncached_tasks_batched = tasks_batch_combined

    # raise AssertionError

    job_manager.run_tasks_batched(task_batch_lst=uncached_tasks_batched, check_cache=False)


import os
import os
import sys
from tabflow.cli.launch_jobs import JobManager
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
# from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
"""
bash ./tabrepo/tabflow/docker/build_docker.sh tabarena tabarena-neerick 763104351884 097403188315 us-west-2
 
aws s3 cp --recursive "s3://prateek-ag/neerick-exp-3/" ../data/neerick-exp-3/ --exclude "*.log"
https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs 
"""

docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/andrej-test:test-csv"
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

batch_sizes = {'APSFailure': 4,
 'Amazon_employee_access': 64,
 'Another-Dataset-on-used-Fiat-500': 128,
 'Bank_Customer_Churn': 128,
 'Bioresponse': 16,
 'Diabetes130US': 64,
 'E-CommereShippingData': 128,
 'Fitness_Club': 128,
 'Food_Delivery_Time': 32,
 'GiveMeSomeCredit': 8,
 'HR_Analytics_Job_Change_of_Data_Scientists': 128,
 'Is-this-a-good-customer': 128,
 'MIC': 128,
 'Marketing_Campaign': 128,
 'NATICUSdroid': 64,
 'QSAR-TID-11': 16,
 'QSAR_fish_toxicity': 128,
 'SDSS17': 8,
 'airfoil_self_noise': 128,
 'anneal': 128,
 'bank-marketing': 64,
 'blood-transfusion-service-center': 128,
 'churn': 128,
 'coil2000_insurance_policies': 128,
 'concrete_compressive_strength': 128,
 'credit-g': 128,
 'credit_card_clients_default': 64,
 'customer_satisfaction_in_airline': 4,
 'diabetes': 128,
 'diamonds': 32,
 'hazelnut-spread-contaminant-detection': 128,
 'healthcare_insurance_expenses': 128,
 'heloc': 128,
 'hiva_agnostic': 16,
 'houses': 64,
 'in_vehicle_coupon_recommendation': 64,
 'jm1': 128,
 'kddcup09_appetency': 64,
 'maternal_health_risk': 128,
 'miami_housing': 32,
 'online_shoppers_intention': 128,
 'physiochemical_protein': 16,
 'polish_companies_bankruptcy': 64,
 'qsar-biodeg': 128,
 'seismic-bumps': 128,
 'splice': 64,
 'students_dropout_and_academic_success': 64,
 'superconductivity': 8,
 'taiwanese_bankruptcy_prediction': 64,
 'website_phishing': 128,
 'wine_quality': 64}

if __name__ == "__main__":
    task_metadata = load_task_metadata()
    task_metadata = load_task_metadata()

    experiment_name = "all_in_one"
    max_concurrent_jobs = 5000
    batch_size = 20
    wait = True
    s3_bucket = "prateek-ag"
    region_name = "us-west-2"
    instance_type = "ml.m6i.2xlarge" #"ml.g6.2xlarge" 

    methods_file = f"configs_all_{experiment_name}.yaml"  # TODO: Need to create this file
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)

    datasets = list(task_metadata["name"])

    # raise AssertionError("No datasets found")

    # toy run
    datasets = ['hiva_agnostic', 'Bioresponse', 'kddcup09_appetency', 'MIC','Diabetes130US', 'anneal']
    folds = list(range(3))
    repeats = list(range(1))
    methods = methods[:100]

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

    tasks_dense = job_manager.get_tasks_dense(
        datasets=datasets,
        repeats=repeats,
        folds=folds,
        methods=methods,
    )
    # uncached_tasks = job_manager.filter_to_only_uncached_tasks(tasks=tasks_dense, verbose=True)

    tasks_batch = job_manager.batch_tasks(tasks=tasks_dense, batch_size=batch_size)
    tasks_batch_combined = tasks_batch

    uncached_tasks_batched = tasks_batch_combined

    # raise AssertionError

    job_manager.run_tasks_batched(task_batch_lst=uncached_tasks_batched, check_cache=False)
import subprocess
import os
import pts_amd as pts

# MLSE library name on Elastic
# library_name = "pts_hipblaslt_benchmark_data-v1.0.0"
library_name = 'pts_rocthrust_benchmark_data-v1.0.0'

# Path for dataset
dataset_path = "/root/repos/hipBLASLt/hipBLASLt_PTS_Benchmarks/"

# Available benchmarks list
benchmarks_list = [
    "matmul_bench",
]

# Path for executable benchmark (new dataset)
# TODO- root folder in /roblemos/
path_executable_new_dataset = "/root/repos/hipBLASLt/build/release/clients/staging/hipblaslt-bench"

# Path for executable benchmark (reference dataset)
# TODO- root folder in /roblemos/
path_executable_reference_dataset = "/root/repos/hipBLASLt/build/debug/clients/staging/hipblaslt-bench"

# Path for script
path_script = "clients/scripts/pts/problems/"

# Create if not existent the dataset_path

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

for benchmark_name in benchmarks_list:
    try:
        # Running the script in the path_executable_new_dataset
        # Create performance results for build_new

        arguments = [
            "clients/scripts/pts/bench_pts_data.py",
            f"{path_executable_new_dataset}",
            f"{dataset_path}",
            "build_new",
            f"{path_script}{benchmark_name}.yaml",
        ]
        path_to_executable = "python3"
        executable = [path_to_executable]

        process = subprocess.check_call(executable + arguments)

    except Exception as err:
        print(err)
        print(
            "There was an error for the new dataset during the performance generation"
        )

    try:
        # Create performance results for build_reference

        arguments = [
            "clients/scripts/pts/bench_pts_data.py",
            f"{path_executable_reference_dataset}",
            f"{dataset_path}",
            "build_reference",
            f"{path_script}{benchmark_name}.yaml",
        ]
        path_to_executable = "python3"
        executable = [path_to_executable]

        process = subprocess.check_call(executable + arguments)

    except Exception as err:
        print(err)
        print(
            "There was an error for the reference dataset during the performance generation"
        )

    # Establishing elastic search connection
    es = pts.establish_connection()

    # Required information and input files for calling create function
    # index_name field on Elastic is the name of the MLSE library
    # index_name for hipBLASLt library is 'pts_hipblaslt_benchmark_data'
    index_name = library_name

    comment = "Ingestion Test from a Local Machine"

    # Inserting files into elastic search index
    # response_code = 1 the data was created successfully in Elastic
    # response_code = 0 the data was not created successfully in Elastic
    # dataset_name corresponds to the name of the dataset created in Elastic
    # (e.g., hipBLASLt_PTS_Benchmarks_matmul_bench_<architecture_name>_<commit-hash>_<YYYYMMDD>_<HH_MM_SS>)

    dataset_path_recorded = f"{dataset_path}hipBLASLt_PTS_Benchmarks_{benchmark_name}/"
    reference_dataset = f"{dataset_path_recorded}build_reference/"
    new_dataset = f"{dataset_path_recorded}build_new/"

    response_code, dataset_name, branch_name = pts.create(
        output_folder_location=dataset_path_recorded,
        index_name=index_name,
        reference_dataset=reference_dataset,
        new_dataset=new_dataset,
        comment=comment,
    )

    if response_code == 1:
        print(f"Dataset {dataset_name} was created successfully in Elastic")
    else:
        print("Dataset was not created successfully in Elastic")

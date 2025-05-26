import os
import random

def generate_data_file(filepath, num_build_points, num_query_points, dimension, max_coord_value=1000):
    """Generates a single data file with specified parameters."""
    with open(filepath, 'w') as f:
        f.write(f"{num_build_points} {num_query_points} {dimension}\n")
        # Generate build points
        for _ in range(num_build_points):
            coordinates = [str(random.randint(0, max_coord_value)) for _ in range(dimension)]
            f.write(" ".join(coordinates) + "\n")
        # Generate query points
        for _ in range(num_query_points):
            coordinates = [str(random.randint(0, max_coord_value)) for _ in range(dimension)]
            f.write(" ".join(coordinates) + "\n")

def create_datasets(base_dir=".", dimensions=[100, 500, 1000], num_files_per_dataset=100,
                    num_build_points=1000, num_query_points=100):
    """Creates directories and generates data files for specified dimensions."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for dim in dimensions:
        dataset_dir = os.path.join(base_dir, f"data_{dim}")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        print(f"Generating dataset in {dataset_dir} for dimension {dim}...")
        for i in range(1, num_files_per_dataset + 1):
            filepath = os.path.join(dataset_dir, f"{i}.txt")
            generate_data_file(filepath, num_build_points, num_query_points, dim)
            if i % 10 == 0: # Print progress every 10 files
                print(f"  Generated {i}/{num_files_per_dataset} files for dim {dim}...")
        print(f"Finished generating dataset for dimension {dim}.")

if __name__ == "__main__":
    # By default, create datasets in the current directory.
    # You might want to change base_dir if you run this script from somewhere else
    # relative to where you want the 'data_100', 'data_500', 'data_1000' folders.
    create_datasets()
    print("All datasets generated successfully.") 
import os
import time
import glob
import json
import sys
sys.path.append("utils")
import transformations
import metrics


# ASSUMPTION: run via shell script, paths will be consistent
input_dir = '/app/input'
output_dir = '/app/output'


def main(input_dir, output_dir):
    start_time = time.time()
    
    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure input dir exists
    try:
        assert os.path.exists(input_dir)
    except AssertionError:
        print(f'ERROR: failed to volume mount input directory')
        return

    # Check if groundtruth file exists
    label_json_path = os.path.join(input_dir, "ground_truth.json")
    try:
        with open(label_json_path, "r") as f:
            label = json.load(f)
    except Exception as e:
        label = None
        print("Evaluation metrics cannot be calculated.")

    # Detect horizon line and write image to output folder
    img_paths = glob.glob(input_dir + "/frame*")
    loss = []
    for input_path in img_paths:
        loss.append(transformations.detectHorizon(input_path, output_dir, label))

    print("Processing done")
    if label:
            print("Printing evaluation metrics...")
            metrics.printEvaluationMetrics(loss)
    time_taken = time.time() - start_time
    print(f'Time Taken to process {len(img_paths)} images is {time_taken}')


if __name__ == '__main__':
    main(input_dir, output_dir)

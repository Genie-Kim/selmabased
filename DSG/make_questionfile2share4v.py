import json
import argparse
import os
from tqdm import tqdm

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset JSON path.')
    parser.add_argument('--questionjsonpath', type=str, default="datasets/ShareGPT4V/data/sharegpt4v/Dict_DSG_nondupid_generated_sharegpt4v_instruct_gpt4-vision_cap100k.json", help='Path to the dataset JSON file')
    parser.add_argument('--exp_root', type=str, default="exp_results", help='Path to the experiment results file')
    parser.add_argument('--exp_path', type=str, required=True, help='Path to the model file')
    return parser.parse_args()

# Main function
def main():
    args = parse_arguments()
    
    sorted_args = sorted(vars(args).items())
    for arg, value in sorted_args:
        # Print argument names in blue and values in green for clear differentiation
        print(f"\033[34m{arg}\033[0m: {value}")
    input("Press Enter to continue...")
    
    json_path = args.questionjsonpath
    exp_path = args.exp_path
    exp_root = args.exp_root
    model_name = exp_path.replace('/','_')

    model_exppath = os.path.join(exp_root, exp_path)
    output_jsonl_path = os.path.join(exp_root,f"{model_name}_questionFORshare4veval.jsonl")
    # Load the JSON data
    with open(json_path, 'r') as file:
        jsondata = json.load(file)
    
    output_jsonlist = []
    # Process the data (example)
    for folder, folderdict in tqdm(jsondata.items()):
        for idx, v in folderdict.items():
            for qid, question in enumerate(v['question_list']):
                tempdict = {}
                tempdict['question_id'] = idx+'##@'+str(qid)
                tempdict['image'] = os.path.join(model_exppath,f"{idx}.jpg")
                tempdict['text']=question['question_natural_language'] + " Answer yes or no."
                tempdict['category']="default"
                tempdict['folder']=folder
                output_jsonlist.append(tempdict)
          
    # Write the output to a JSONL file
    with open(output_jsonl_path, 'w') as outfile:
        for entry in output_jsonlist:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    main()

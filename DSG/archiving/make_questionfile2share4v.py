import json
import argparse
import os
from tqdm import tqdm

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset JSON path.')
    parser.add_argument('--questionjsonpath', type=str, default="datasets/docci/docci_metadata.jsonlines", help='Path to the dataset JSON file')
    parser.add_argument('--exp_root', type=str, default="exp_results", help='Path to the experiment results file')
    parser.add_argument('--exp_path', type=str, required=True, help='Path to the model file')
    parser.add_argument(
    "--splits",
    nargs="+",
    type=str,
    choices=[
        "train",
        "test",
        "qual_test",
        "qual_dev",
    ],
    default=["test","qual_test","qual_dev"],
)

    return parser.parse_args()

# Main function
def main():
    args = parse_arguments()
    
    sorted_args = sorted(vars(args).items())
    for arg, value in sorted_args:
        # Print argument names in blue and values in green for clear differentiation
        print(f"\033[34m{arg}\033[0m: {value}")
    input("Press Enter to continue...")
    
    model_name = args.exp_path.replace('/','&')

    model_exppath = os.path.join(args.exp_root, args.exp_path)
    output_jsonl_path = os.path.join(args.exp_root,f"{model_name}&questionFORshare7Beval.jsonl")
    # Load the JSON data
    docci_id2meta={}
    with open(args.questionjsonpath, "r") as f:
        for line in f:
            item = json.loads(line)
            if 'dsg' in item:
                key = item.pop("example_id")
                docci_id2meta[key] = item
            
    output_jsonlist = []
    # Process the data (example)
    for example_id, values in tqdm(docci_id2meta.items()):
        split = '_'.join(example_id.split('_')[:-1])
        image_path = os.path.join(model_exppath,split,f"{example_id}.jpg")
        assert os.path.isfile(image_path), f"there is no image file in {image_path}"
        
        if split in args.splits:
            question_all_sentences = values['dsg']['question']
            for sentence_id, ques_info in question_all_sentences.items():
                questions = ques_info['question']
                for question_id, question in questions.items():
                    tempdict = {}
                    tempdict['question_id'] = f"{example_id}&{sentence_id}&{question_id}"
                    tempdict['image'] = image_path
                    tempdict['text']=f"{question} Answer yes or no."
                    tempdict['category']="default"
                    output_jsonlist.append(tempdict)
            
    # Write the output to a JSONL file
    with open(output_jsonl_path, 'w') as outfile:
        for entry in output_jsonlist:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    main()

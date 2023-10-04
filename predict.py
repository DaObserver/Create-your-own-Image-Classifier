import argparse
import torch
import numpy as np
import json
import fmodel

parser = argparse.ArgumentParser(description='Parser for predict.py')
parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
parser.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu
json_name = args.category_names
path = args.checkpoint

def main():
    model = fmodel.load_checkpoint(path)
    
    with open(json_name, 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    probabilities, classes = fmodel.predict(path_image, model, number_of_outputs, device)
    
    for prob, class_idx in zip(probabilities, classes):
        class_label = cat_to_name[str(class_idx)]
        print(f"Class: {class_label}, Probability: {prob:.4f}")
    
if __name__ == "__main__":
    main()

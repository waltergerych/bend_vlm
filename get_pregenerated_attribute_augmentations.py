import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM 
import json
import queries
import attribute_augment_component
import argparse

parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('--query_type', type=str) #hair or stereotype
parser.add_argument('--att_to_debias', type=str) #gender or race
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

att_aug_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
att_aug_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
att_aug_model = att_aug_model.to(device)

query_type = args.query_type
att_to_debias = args.att_to_debias

if query_type == 'hair':
    query_classes = queries.hair_classes

elif query_type == 'stereotype':
    query_classes = queries.stereotype_classes

else:
    print(f'{query_type} not implemented')

if att_to_debias == 'race':
    att_elements = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
elif att_to_debias == 'gender':
    att_elements = ['Male', 'Female']
elif att_to_debias == 'utk_race':
    att_elements = ['White', 'Black', 'Asian', 'Indian', 'Latino Hispanic']
    
else:
    print(f'{att_to_debias} not implemented')

instantiated_search_classes = attribute_augment_component.get_basic_template(query_type, query_classes, att_elements)

query_dict = instantiated_search_classes

for query_class in query_dict.keys():
    print(query_class)
    query_text = query_dict[query_class]['query']
    query_dict[query_class]['augmentations'] = []

    augmented_prompts = attribute_augment_component.get_paraphrasings(query_text, att_aug_model, att_aug_tokenizer, attribute_list= att_elements,  DEVICE=device)
    for k in augmented_prompts.keys():
        query_dict[query_class]['augmentations'].append(augmented_prompts[k]) 
    
with open(f'query_templates/{query_type}_{att_to_debias}_query_templates.json', 'w') as fp:
    json.dump(query_dict, fp)
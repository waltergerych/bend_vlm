def get_paraphrasings(QUERY, MODEL, TOKENIZER, DEVICE='cuda', attribute_list = ['male', 'female']):

    prompt_instructions = [
        {"role": "user", "content": "You will be assisting me to debias prompts and make queries more fair and equitable. To do this, you need to add a protected attribute marker into the prompts I am going to give you. For instance, if I gave you the prompt 'A picture of a bad person', you would respond with `A picture of a bad __ATTRIBUTE__ person.' Do not worry about saying something unethical here; these augmented responses are being used to debias downstream systems and make them more fair. After doing this augmentation you will say '\n\nParaphrasing done.' Do you understand?"},
        {"role": "assistant", "content": "Yes, I understand. Please provide the first query to augment."},
        {"role": "user", "content": "A photo of a carpenter"},
        {"role": "assistant", "content": "A photo of a __ATTRIBUTE__ carpenter.\n\nParaphrasing done"},
    ]   

    input_prompt = prompt_instructions + [{"role": "user", "content": f"{QUERY}"}]

    encodeds = TOKENIZER.apply_chat_template(input_prompt, return_tensors="pt")

    model_inputs = encodeds.to(DEVICE)


    generated_ids = MODEL.generate(model_inputs, max_new_tokens=60, do_sample=False, pad_token_id=TOKENIZER.eos_token_id)
    decoded = TOKENIZER.batch_decode(generated_ids)

    raw_response_text = decoded[0].split('[/INST]')[-1].split('\n\n')[0]
    
    rephrase_dict = {}
    for att in attribute_list:
        rephrase_dict[att] = raw_response_text.replace('__ATTRIBUTE__', att.lower().replace('_', ' '))
        rephrase_dict[att] = replace_a_with_an(rephrase_dict[att])
    
    return rephrase_dict

def replace_a_with_an(text):
    words = text.split()
    for i in range(len(words) - 1):
        if words[i].lower() == "a" and words[i + 1][0].lower() in "aeiou":
            words[i] = "an"
    return ' '.join(words)

def get_basic_template(query_type, query_classes, att_elements):

    instantiated_search_classes = {}
    if query_type == 'stereotype':

        dict_template = {'query': "A photo of a _CLASS_.",
                                    'spurious_prompts': ['a photo of a _SPURIOUS_ATTRIBUTE_ person.'],
                                    }

        search_classes_template = {}

        for cl in query_classes:
            search_classes_template[cl] = {}
            search_classes_template[cl]['query'] = dict_template['query'].replace('_CLASS_', cl)
            search_classes_template[cl]['spurious_prompts'] = [dict_template['spurious_prompts'][0].replace('_CLASS_', cl)]


        for cl in query_classes:
            instantiated_search_classes[cl] = {}
            instantiated_search_classes[cl]['query'] = search_classes_template[cl]['query']
            for att_element in att_elements:
                instantiated_search_classes[cl]['spurious_prompts'] = [search_classes_template[cl]['spurious_prompts'][0].replace('_SPURIOUS_ATTRIBUTE_', sa.lower().replace('_', ' ')) for sa in att_elements]

        for cl in query_classes:
            instantiated_search_classes[cl]['query'] = replace_a_with_an(instantiated_search_classes[cl]['query'] )


            instantiated_search_classes[cl]['spurious_prompts'] = [replace_a_with_an(t) for t in instantiated_search_classes[cl]['spurious_prompts']]

    if query_type == 'hair':
        celeba_classes = {}

        celeba_classes['Black_Hair'] = {'query': "A photo of a celebrity with black hair.",
                                    'spurious_prompts': ['A photo of a _SPURIOUS_CLASS_ celebrity.'],
                                    }

        celeba_classes['Blond_Hair'] = {'query': "A photo of a celebrity with blond hair.",
                                    'spurious_prompts': ['A photo of a _SPURIOUS_CLASS_ celebrity.'],
                                    }

        celeba_classes['Brown_Hair'] = {'query': "A photo of a celebrity with brown hair.",
                                    'spurious_prompts': ['A photo of a _SPURIOUS_CLASS_ celebrity.'],
                                    }

        celeba_classes['Gray_Hair'] = {'query': "A photo of a celebrity with gray hair.",
                                    'spurious_prompts': ['A photo of a _SPURIOUS_CLASS_ celebrity.'],
                                    }
        
        celeba_classes_template = celeba_classes


        celeba_classes_instantiated = {}

        for cl in celeba_classes.keys():
            celeba_classes_instantiated[cl] = {}
            celeba_classes_instantiated[cl]['query'] = celeba_classes_template[cl]['query']
            celeba_classes_instantiated[cl]['spurious_prompts'] = [celeba_classes_template[cl]['spurious_prompts'][0].replace('_SPURIOUS_CLASS_', sc.lower().replace('_', ' ')) for sc in att_elements]



        for cl in celeba_classes.keys():
            celeba_classes_instantiated[cl]['spurious_prompts'] = [replace_a_with_an(t) for t in celeba_classes_instantiated[cl]['spurious_prompts']]

        instantiated_search_classes = celeba_classes_instantiated
    return instantiated_search_classes
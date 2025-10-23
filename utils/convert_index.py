import json

def modify_value(value):
    if value.startswith("<a_"):
        return int(value[3:-1])
    elif value.startswith("<b_"):
        return int(value[3:-1]) + 512
    elif value.startswith("<c_"):
        return int(value[3:-1]) + 512 * 2
    elif value.startswith("<d_"):
        return int(value[3:-1]) + 512 * 3
    return value

def modify_json_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    modified_data = {}
    for index, items in data.items():
        new_index = str(int(index) + 1)  
        modified_data[new_index] = [modify_value(item) for item in items]
    
    with open(output_file, 'w') as outfile:
        json.dump(modified_data, outfile, indent=2)

input_file = 'merge5-v2.index.json'  
output_file = 'convert_merge5-v2.json'  

modify_json_file(input_file, output_file)

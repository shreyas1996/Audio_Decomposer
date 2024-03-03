import json
import random

# Number of elements to select
num_elements = 50  # Change this to the number of elements you want

# Load the data from the original JSON file
with open('assets/train_files_list.json', 'r') as f:
    train_data = json.load(f)

# Load the data from the original JSON file
with open('assets/test_files_list.json', 'r') as f:
    test_data = json.load(f)

# Randomly select 'num_elements' elements from the list
selected_train_data = random.sample(train_data, num_elements)
selected_test_data = random.sample(test_data, num_elements)

# Write the selected elements to a new JSON file
with open('assets/small_train_files_list.json', 'w') as f:
    json.dump(selected_train_data, f)
with open('assets/small_test_files_list.json', 'w') as f:
    json.dump(selected_test_data, f)
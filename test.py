import os
import re



BASE_DIR = 'checkpoints_lambda_1'


def extract_numbers_from_files(directory, max_cbf=None, max_action=None):
    for filename in os.listdir(directory):
        match_cbf = re.search(r"cbf_net_step_(\d+)", filename)
        match_action = re.search(r"action_net_step_(\d+)", filename)
        if match_cbf:
            number = int(match_cbf.group(1))
            print(f"File: {filename}, Number: {number}")
            max_cbf = max(max_cbf, number) if max_cbf else number
        if match_action:
            number = int(match_action.group(1))
            print(f"File: {filename}, Number: {number}")
            max_action = max(max_action, number) if max_action else number
        
    return max_cbf, max_action


max_cbf, max_action = extract_numbers_from_files(BASE_DIR)

print(f"Max CBF: {max_cbf}")
print(f"Max Action: {max_action}")

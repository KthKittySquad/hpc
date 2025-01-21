import re
from collections import defaultdict

with open('timing_output.txt', 'r') as f:
    data = f.read()

function_pattern = r"Function: (.+)"
time_pattern = r"Average Execution Time: (\d+\.\d+)"
std_dev_pattern = r"Standard Deviation: (\d+\.\d+)"
function_data = defaultdict(list)

functions = re.findall(function_pattern, data)
times = re.findall(time_pattern, data)
std_devs = re.findall(std_dev_pattern, data)

# Ensure that we have a matching number of functions, times, and standard deviations
if len(functions) == len(times) == len(std_devs):
    for function, time, std_dev in zip(functions, times, std_devs):
        function_data[function].append((float(time), float(std_dev)))
    for function, records in function_data.items():
        avg_exec_time = sum(record[0] for record in records) / len(records)
        avg_std_dev = sum(record[1] for record in records) / len(records)
        print(f"Function: {function}")
        print(f"  Average Execution Time: {avg_exec_time:.6f} seconds")
        print(f"  Average Standard Deviation: {avg_std_dev:.6f} seconds")
else:
    print("Data mismatch: The number of functions, times, and standard deviations do not match.")

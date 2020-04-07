# To be placed in ILSVRC2012_img_val, alongside the "classification" and "otherClass" folders

import csv
import os

'''
# Make test csvs
thing1 = {1: "This", 2: "That"}
thing2 = {3: "foo", 4: "bar"}

with open("part1.csv", "w+") as f:
    writer = csv.writer(f, lineterminator='\n')
    for row in thing1.items():
        writer.writerow(row)

with open("part2.csv", "w+") as f:
    writer = csv.writer(f, lineterminator='\n')
    for row in thing2.items():
        writer.writerow(row)

files = ["part1.csv", "part2.csv"]

with open("full.csv", "w+") as full:
    writer = csv.writer(full, lineterminator='\n')
    for f_path in files:
        with open(f_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                writer.writerow(row)
'''

# Manually, because the order matters
# Device	    Start	End
# Desktop	    1	    10000
# gcp-cmpe351-1	10001	20000
# gcp-cmpe351-2	20001	30000
# AWS_Correct	30001	35000
# AWS_Correct_2	35001	40000
# XPS-15	    40001	50000

main = "classification/"
base = "otherClass/"
dirs = ["Desktop_Classify/", "gcp_1_Classify/", "gcp_2_Classify/", "AWS_Correct_Classify/", "AWS_Correct_2_Classify/", "XPS15_Classify/"]

# Combine csvs together, with minimal memory usage
# (Unneccesary in this case, the files are small, but whatever)

for d in dirs:
    file_names = os.listdir(base + d)
    for file_name in file_names:
        with open(main + file_name, "a") as full:
            writer = csv.writer(full, lineterminator='\n')
            with open(base + d + file_name, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    writer.writerow(row)
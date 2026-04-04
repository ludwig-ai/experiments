import sys
from datetime import datetime
from time import mktime

# Produce csv of autoscaler add/remove nodes for hyperparameter search

# Run with 2 arguments:
# 1) Unix time of first trial start (output from trial_sequence_map.py)
# 2) Preprocessed autoscaler logs (output of grepaddsubtract.sh)

job_start_time_unix = float(sys.argv[1])
filename = sys.argv[2]

ADD_NODE_SUFFIX = " is newly setup, treating as active\n"
REM_NODE_SUFFIX = ".\n"

node_dict = {}

file = open(filename, 'r')
while True:
    line = file.readline()
    if not line:
        break
    time_stamp = line.strip("example-cluster,default:")[:19]
    time_stamp_unix = mktime(datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S").timetuple())
    if time_stamp_unix >= job_start_time_unix:
        if line.endswith(ADD_NODE_SUFFIX):
            node_added = line.strip(ADD_NODE_SUFFIX).split(' ')[-1]
            if node_added not in node_dict:
                node_dict[node_added] = {'start_time': time_stamp_unix}
            else:
                print("TODO handle added recycled address:", node_added, time_stamp, time_stamp_unix)
        elif line.endswith(REM_NODE_SUFFIX):
            node_removed = line.strip(REM_NODE_SUFFIX).split(' ')[-1]
            if node_removed in node_dict:
                node_duration = time_stamp_unix - node_dict[node_removed]['start_time']
                node_dict[node_removed]['duration'] = node_duration
            else:
                print("Ignoring pre-run node removed:", node_removed, time_stamp, time_stamp_unix)
        else:
            print("Ignoring unrecognized line:",line)

# Output node usage sorted by earliest starttime
with open('nodeupdown.csv', 'w') as f:
    print("node,starttime,runtime", file=f)
    for node_key in node_dict:
        print("{0},{1},{2}".format(
            node_key, node_dict[node_key]['start_time']-job_start_time_unix, node_dict[node_key]['duration']), file=f)

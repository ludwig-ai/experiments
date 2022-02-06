import sys
from datetime import datetime
from time import mktime

# Produce csv files:
# 1) trials with start,duration,node
# 2) nodes with start,duration,trial
# 3) nodes summary with start,duration,end

# Run on hyperparameter search log list; each log stored in dataset dir
# Run from at least one directory above dataset dir to get dataset name

trial_dict = {}
min_trial_start_time = None
for index in range(1, len(sys.argv)):
    filename = sys.argv[index]
    dirs = filename.split('/')
    if len(dirs) > 1:
        dataset_name = dirs[-2]
    else:
        dataset_name = None
    hrs = filename[-3:]

    file = open(filename, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        if line.startswith("Current time: "):
            last_current_time = line.strip("Current time: ")[:-27]
            last_current_time_unix = mktime(datetime.strptime(last_current_time, "%Y-%m-%d %H:%M:%S").timetuple())
        if line.startswith("| trial"):
            cols = line.split('|')
            trial_name = cols[1].strip()
            trial_status = cols[2].strip()
            trial_location = cols[3].strip().split(':')[0]
            if trial_name not in trial_dict and trial_status == 'RUNNING':
                trial_dict[trial_name] = {'loc': trial_location, 'start_time': last_current_time_unix, 'hours': hrs}
                if dataset_name is not None:
                    trial_dict[trial_name]['dataset'] = dataset_name
                if min_trial_start_time == None or last_current_time_unix < min_trial_start_time:
                    min_trial_start_time = last_current_time_unix
            if trial_name in trial_dict and trial_status == 'TERMINATED':
                if 'duration' not in trial_dict[trial_name]:
                    trial_duration = cols[16].strip()
                    if len(trial_duration) == 0:
                        trial_dict[trial_name]['duration'] = last_current_time_unix - trial_dict[trial_name]['start_time']
                    else:
                        trial_dict[trial_name]['duration'] = trial_duration

print("Unix time of first trial start:", min_trial_start_time)

# Output the trial duration information
location_dict = {}
with open('trialruns.csv', 'w') as f:
    print("trial,starttime,runtime,location", file=f)
    for key in trial_dict:
        if 'duration' in trial_dict[key]:
            duration = trial_dict[key]['duration']
            if duration != '0':
                location = trial_dict[key]['loc']
                start_offset = trial_dict[key]['start_time'] - min_trial_start_time
                if 'dataset' not in trial_dict[key]:
                    trial_key = trial_dict[key]['hours'] + "/" + key
                else:
                    trial_key = trial_dict[key]['dataset'] + "/" + trial_dict[key]['hours'] + "/" + key
                print("{0},{1},{2},{3}".format(trial_key, start_offset, duration, location), file=f)
                if location not in location_dict:
                    location_dict[location] = []
                location_dict[location].append({'starttime': start_offset, 'runtime': duration, 'trial': trial_key})

location_order_dict = {}
for loc_key in location_dict:
    sorted_start = sorted(location_dict[loc_key], key = lambda i: i['starttime'])
    location_dict[loc_key] = sorted_start
    location_order_dict[loc_key] = location_dict[loc_key][0]['starttime']

# Output the node usage information
with open('nodeuse.csv', 'w') as g, open('nodeusesummary.csv', 'w') as h:
    print("location,starttime,runtime,trial", file=g)
    print("location,starttime,runtime,endtime", file=h)
    for loc_key in sorted(location_order_dict, key=location_order_dict.__getitem__):
        first_use_time = None
        last_use_time = None
        for loc_val in location_dict[loc_key]:
            print("{0},{1},{2},{3}".format(loc_key, loc_val['starttime'], loc_val['runtime'], loc_val['trial']), file=g)
            start_time = float(loc_val['starttime'])
            end_time = start_time + float(loc_val['runtime'])
            if first_use_time is None:
                first_use_time = start_time
            if last_use_time is None or end_time > last_use_time:
                last_use_time = end_time
        print("{0},{1},{2},{3}".format(loc_key, first_use_time, last_use_time-first_use_time, last_use_time), file=h)

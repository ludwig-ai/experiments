import sys
import json

hyperopt_stats = {}
with open(sys.argv[1]) as json_file:
    hyperopt_stats = json.load(json_file)

metric = hyperopt_stats["hyperopt_config"]["metric"]
output_feature = hyperopt_stats["hyperopt_config"]["output_feature"]

metric_score = hyperopt_stats["hyperopt_results"][0]["metric_score"]
parameters = hyperopt_stats["hyperopt_results"][0]["parameters"]

vali_stats = hyperopt_stats["hyperopt_results"][0]["training_stats"]["validation"]
combined_losses = vali_stats[output_feature][metric]
ind_offset = -1
for ind in range(len(combined_losses)):
    if combined_losses[ind] == metric_score:
        ind_offset = ind
        break
print("best metric_score offset in validation", ind_offset)
print("best metric_score hyperparameters", parameters)

test_stats = hyperopt_stats["hyperopt_results"][0]["training_stats"]["test"]
test_loss = test_stats[output_feature][metric][ind_offset]
print("offset metric_score in test ", test_loss)

print("offset metric_score in vali ", metric_score)

train_stats = hyperopt_stats["hyperopt_results"][0]["training_stats"]["training"]
train_loss = train_stats[output_feature][metric][ind_offset]
print("offset metric_score in train", train_loss)

metric_of_interest = sys.argv[2]
for key in test_stats.keys():
    if key != "combined":
        test_acc = test_stats[key][metric_of_interest][ind_offset]
        print(metric_of_interest, "offset in test ", key, test_acc)
        vali_acc = vali_stats[key][metric_of_interest][ind_offset]
        print(metric_of_interest, "offset in vali ", key, vali_acc)
        train_acc = train_stats[key][metric_of_interest][ind_offset]
        print(metric_of_interest, "offset in train", key, train_acc)

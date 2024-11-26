import matplotlib
import matplotlib.pyplot as plt


import os

# The big function.
# directory is mandatory, must be the path to a folder uniformly consisting of log files
# Color can be default selected by matplotlib, or you can specify
# Scale is used to correct old logs, which sometimes have the halving error
# Splitter is used for inconsistent logs, which sometimes use ', '
def add_dir_to_plot(directory, color=None, scale=1, splitter='; '):
  cost_vals = []
  acc_vals = []
  file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

  for file_name in file_names:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
      # Currently, all non-baselines use the same log format. 
      # This assumes the final average is found on the last line -- which it should be if the lines are written right
      last_line_segments = file.readlines()[-1].strip().split(splitter)[6:8]
      cost_vals.append(float(last_line_segments[1]) * scale)
      acc_vals.append(float(last_line_segments[0]) * scale)

  plt.scatter(cost_vals, acc_vals, c=color)

# The actual inclusions into the data
# add_dir_to_plot('simulation/logs/tanhexpsupervised', 'purple', splitter=', ')
# add_dir_to_plot('simulation/logs/expsupervised', 'blue', splitter='; ')
# add_dir_to_plot('simulation/logs/oldsupervised', 'green', scale=2, splitter=', ')
# add_dir_to_plot('simulation/logs/500neural', 'green')

# add_dir_to_plot('simulation/logs/testsupervised', 'blue')
# add_dir_to_plot('simulation/logs/testmufasa', 'green')
add_dir_to_plot('simulation/logs/limmufasa', 'purple')

# Baseline Models
baseline_cost = []
baseline_acc = []

with open('simulation/logs/baseline.txt', 'r') as baseline:
  for line in baseline:
    segments = line.strip().split('; ')
    baseline_cost.append(float(segments[2]))
    baseline_acc.append(float(segments[1]))

# Plot all points
plt.scatter(baseline_cost, baseline_acc, c='red', marker='x')

# Plot linear interpolation of baselines
cost_and_acc = sorted(zip(baseline_cost, baseline_acc))

# Follows best cost at time accuracy
bests_at_cost = [cost_and_acc[0]]
for cost, acc in cost_and_acc[1:]:
  if acc >= bests_at_cost[-1][1]:
    if cost == bests_at_cost[-1][0]:
      bests_at_cost.pop(-1)
    bests_at_cost.append((cost, acc))
for i in range(len(bests_at_cost) - 1):
  plt.plot([bests_at_cost[i][0], bests_at_cost[i+1][0]], [bests_at_cost[i][1], bests_at_cost[i+1][1]], c='red', marker='', linestyle=':')




# Dimensions, labeling, saving
plt.xlabel("Cost")
plt.ylabel("Accuracy")
# plt.ylim((.98, .995))
# plt.xlim((50, 100))
plt.title("Contextual against Baseline")
plt.savefig('visualizations/comp.png')

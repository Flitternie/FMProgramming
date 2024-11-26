import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# 3D Visualizer
# Pulls a log and sees training over time
# Not super useful except to see how quickly the model progresses
time = []
accs = []
costs = []

# Change the filename as necessary
filename = 'simulation/logs/testmufasa/0..txt' 

with open(filename, 'r') as file:
  for line in file:
    line_segments = line.strip().split('; ')[6:8]
    acc = float(line_segments[0])
    cost = float(line_segments[1]) / 100 # 100 from the max cost on benchmark
    accs.append(acc)
    costs.append(cost)

time = [i / len(accs) for i in range(len(accs))]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(time, accs, costs, c='blue')
ax.set_xlabel('time')
ax.set_ylabel('accuracy')
ax.set_zlabel('cost')
ax.set_title("Weight: " + filename.split('/')[-1][:-4])
# plt.savefig("Weight: " + filename.split('/')[-1][:-4] + " prog.png") # Saving isn't very useful
plt.show()


# Scatterplot over training time
# Uses f1 metric, which doesn't 100% make sense, but kinda works
# Basically discard the first 10% of time
# with open(filename, 'r') as file:
#   for line in file:
#     line_segments = line.strip().split('; ')[6:8]
#     acc = float(line_segments[0])
#     cost = float(line_segments[1]) / 100 # 100 from the max cost on benchmark
#     f1.append(2 * acc * cost / (acc + cost))

# plt.scatter([i / len(f1) for i in range(1, len(f1) + 1)], f1, c='blue')

# Change the filename as necessary
# filename = 'simulation/logs/expsupervised/0.05.txt'

# time = []
# f1 = []

# with open(filename, 'r') as file:
#   for line in file:
#     line_segments = line.strip().split('; ')[6:8]
#     acc = float(line_segments[0])
#     cost = float(line_segments[1]) / 100 # 100 from the max cost on benchmark
#     f1.append(2 * acc * cost / (acc + cost))

# plt.scatter([i / len(f1) for i in range(1, len(f1) + 1)], f1, c='purple')

# plt.xlabel("Example Number")
# plt.ylabel("Accuracy-Cost f1")
# plt.ylim((.5, .65))
# plt.xlim((.05, 1))
# # plt.title(filename)
# plt.savefig('visualizations/progcomp.png')


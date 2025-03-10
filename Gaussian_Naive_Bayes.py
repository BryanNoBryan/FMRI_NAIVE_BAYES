import scipy.io
import numpy as np
import math

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return math.sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
# * upacks the dataset for zip to work
def summarizeDataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	return summaries

# # 28 subject, 6 temperatures, 32 inputs (8*4)
# hrd_data = temp_data
def separateByClass(data: list[list[list[float]]]) -> dict[float, list[list[float]]]:
    separated = dict()
    for sub in range(len(data)):
        vector2d = data[sub]
        for temp in range(len(vector2d)):
            class_value = temp
            if class_value not in separated:
                separated[class_value] = list()
            separated[class_value].append(vector2d[temp])
    return separated

# Split dataset by class then calculate statistics for each row
# Contents: dict[temp, (mean, stdev, count)]
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarizeDataset(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, count = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# the fraction of the data to be used to test the model
TEST_FRACTION = 0.1

COMPONENTS = 4
TEMPERATURES = 6
TIMEBINS = 8
SUBJECTS = 28

# make the continous range discrete
DISCRETENESS = 0.00001

mat_data = scipy.io.loadmat('HRD_ALL.mat')

keys = mat_data.keys()
print(keys)

# print(mat_data["__header__"])
# print(mat_data["__version__"])
# print(mat_data["__globals__"])
# print(mat_data["HRD_all"])

# 4D matrix dimensions: 4   6   8  28
# 4 components, 6 temperatures, 8 timebins, 28 subject
hrd_data = mat_data["HRD_all"]
hrd_data = np.array(hrd_data)

hrd_data = np.swapaxes(hrd_data, 3, 0)
# 28 subject, 6 temperatures, 8 timebins, 4 components 

# temporary holder to be placed back as hrd_data
temp_data = np.zeros((SUBJECTS, TEMPERATURES, COMPONENTS*TIMEBINS))

# Align COMPONENTS*TIMEBINS together
for component in range(COMPONENTS):
    value = hrd_data[:, :, :, component]
    temp_data[:, :, component*TIMEBINS:(component+1)*TIMEBINS] = value

# 28 subject, 6 temperatures, 32 inputs (8*4)
hrd_data = temp_data
print("HRD Shape: " + str(hrd_data.shape))

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

frac = round(SUBJECTS*TEST_FRACTION)
print(frac)

# training and testing sets of data

# Using cross-validation
hrd_train = hrd_data[1:(SUBJECTS-1-frac), :, :]
hrd_test = hrd_data[(SUBJECTS-frac):SUBJECTS-1, :, :]
# Just use the whole dataset
# hrd_train = hrd_data[0:(SUBJECTS-1), :, :]
# hrd_test = hrd_data[(0):SUBJECTS-1, :, :]

print('hrd_train dim: ' + str(hrd_train.shape))
print('hrd_test dim: ' + str(hrd_test.shape))


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Gaussian Naive Bayes begins

# dict[temp, [(mean, stdev, len)]]
# number of (mean, stdev, len) is 24, matching our number of inputs
summaries = summarizeByClass(hrd_train)

trials = 0
correct = 0

for sub in range(len(hrd_test)):
    for temp in range(TEMPERATURES):
        trials += 1
        probabilities = calculate_class_probabilities(summaries, hrd_test[sub][temp])
        print([np.format_float_scientific(probabilities[i], precision=3) for i in probabilities])
        # Find the index of the maximum element
        prediction = np.argmax(probabilities)
        
        print("prediction: " + str(prediction))
        if temp == prediction: 
            correct += 1

print("Trials: " + str(trials))
print("Correct: " + str(correct/trials))

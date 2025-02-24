import requests
import pandas as pd
import scipy.io
import numpy as np
import math

import LogSumExp 

# run using: pipenv run python main.py

# the fraction of the data to be used to test the model
TEST_FRACTION = 0.2

COMPONENTS = 4
TEMPERATURES = 6
TIMEBINS = 8
SUBJECTS = 28

# make the continous range discrete
DISCRETENESS = 0.00001

# all the hrd values seen in the training data 
seen_values = set()

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

# get unique list of possible inputs
for sub in range(SUBJECTS):
    for temp in range(TEMPERATURES):
        for i in range(COMPONENTS*TIMEBINS):
            data = hrd_data[sub, temp, i]

            # make the continous range discrete
            data = round(data/DISCRETENESS)
            hrd_data[sub, temp, i] = data

            # append unseen data to a set
            if data not in seen_values:
                seen_values.add(data)

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
# Naive Bayes begins

# P(T-i)
# Calculate probablity(proportion) of each temperature occuring
# well, in our case, we have temp HDR for all patients, so probability is the same...
val_count = SUBJECTS*TEMPERATURES
P_temp = []
for temp in range(TEMPERATURES):
    P_temp.append(1/6)

# Number of inputs in each temperature, well they're all equal in this caase
temp_count = []
for temp in range(TEMPERATURES):
    temp_count.append(SUBJECTS*COMPONENTS*TIMEBINS)

# P(input-i | T-i)
# Calculate probability of input given temperature

# array holding each dictionary that holds probabilities
# the key is the input, the value is its frequency
prob_conditional_freq = []
# same array but scaled into probabilites
prob_conditional = []
for temp in range(TEMPERATURES):
    prob_conditional_freq.append(dict.fromkeys(seen_values, 0))
    prob_conditional.append(dict.fromkeys(seen_values, 0))

# print(seen_values)

# Calculating prob_conditional_freq values
for temp in range(TEMPERATURES):
    for sub in range(SUBJECTS):
        for i in range(COMPONENTS*TIMEBINS):
            data = hrd_data[sub, temp, i]
            
            # Add 1 freqency to this particular input
            prob_conditional_freq[temp][data] += 1

# Calculating prob_conditional values
for temp in range(TEMPERATURES):
    for input in seen_values:
        # Laplace smoothing
        prob_conditional[temp][input] = (prob_conditional_freq[temp][input]+1)/(temp_count[temp]+2)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Testing the model

correct = 0

cases = hrd_test.shape[0]*TEMPERATURES
total_inputs = cases*COMPONENTS*TIMEBINS * (1 + 6)
skipped_inputs = 0

# Logarthimic floating point underflow protection
# So using log instead, addition instead of multiplication


print("Original_Temp  Predicted_Temp")

# loop over all inputs
for sub in range(hrd_test.shape[0]):
    for temperature in range(TEMPERATURES):
        # Start of one testing "unit"

        # Empty list, 6 zeros
        Estimation_P_Temp = [0] * TEMPERATURES
        # look over all temperatures to find probabilities of each
        for temp in range(TEMPERATURES):
            inputs = hrd_test[sub, temp, :]

            probability = 0

            NAIVE_BAYES_NUM = []

            # Baye's formula in logarithm form started
            #NUMERATOR ~~~~~~~~
            NAIVE_BAYES_NUM.append(P_temp[temp])
            NAIVE_BAYES_NUM.append([])

            # loop over each input to remove them if they don't exist in training per temperature
            for i in range(len(inputs)):
                try:
                    hdr = inputs[i]
                    partial_prob = prob_conditional[temp][hdr]
                    NAIVE_BAYES_NUM[1].append(partial_prob)
                # unseen data, remove it
                except KeyError:
                    NAIVE_BAYES_NUM[1].append(1 / temp_count[temp])
                    skipped_inputs += 1
                    print('skipped warning \n\n skipped warning \n\n skipped warning')
            #NUMERATOR ~~~~~~~~

            # use Log-Sum-Exp Trick

            #DENOMINATOR ~~~~~~
            NAIVE_BAYES_DENOM = []

            for temp2 in range(TEMPERATURES):
                NAIVE_BAYES_DENOM.append([])
                NAIVE_BAYES_DENOM[temp2].append(P_temp[temp2])
                NAIVE_BAYES_DENOM[temp2].append([])

                for i in range(len(inputs)):
                    try:
                        hdr = inputs[i]
                        partial_prob = prob_conditional[temp2][hdr]
                        NAIVE_BAYES_DENOM[temp2][1].append(partial_prob)
                    # unseen data, remove it
                    except KeyError:
                        NAIVE_BAYES_DENOM[temp2][1].append(1 / temp_count[temp2])
                        skipped_inputs += 1
                        print('skipped warning \n\n skipped warning \n\n skipped warning')
            #DENOMINATOR ~~~~~~

            print(f'temp:{temp} ', end="")
            probability = LogSumExp.sum_log_exp_naive_bayes(NAIVE_BAYES_NUM, NAIVE_BAYES_DENOM)

            Estimation_P_Temp[temp] = probability

        # print(Estimation_P_Temp)
        confidence = max(Estimation_P_Temp)
        predicted_temp = Estimation_P_Temp.index(max(Estimation_P_Temp))

        WE_DID_IT_YAY = False
        if (temperature == predicted_temp):
            WE_DID_IT_YAY = True
            correct += 1
        
        print(f"~~~~~~~~~case {sub*TEMPERATURES+temperature+1}: orig:{temperature} pred:{predicted_temp}   confidence: {round(confidence,8)}", end="")
        print([np.format_float_positional(i, precision=3) for i in Estimation_P_Temp])
        print('\n')
        # End of one testing "unit"

print(f'{correct} correct / {cases} cases')
print(f'{round(((correct / cases) * 100),ndigits=2)}% accuracy')

print("total inputs:" + str(total_inputs))
print("skipped inputs:" + str(skipped_inputs))
print("%% inputs skipped:" + str(round(((skipped_inputs / total_inputs) * 100),ndigits=2)))
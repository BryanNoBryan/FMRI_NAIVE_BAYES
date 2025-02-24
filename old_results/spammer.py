#v2
#ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#import pandas
import pandas as pd

#Auxillary
import math

#preprocessing
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('wordnet')

#too large takes some time to run
TRAIN_AND_TEST_SIZE = 1000
TEST_FRACTION = 0.1

data = pd.read_csv("spam.csv")
data.drop(data.columns[[0, 1]], axis=1, inplace=True)
data = data.iloc[:TRAIN_AND_TEST_SIZE]

tokenized_list = []

# Removing punctuations in string
# Using regex
for i in range(len(data)):
    #remove punctuation
    data.iloc[i, 0] = re.sub(r'[^\w\s]', '', data.iloc[i, 0])
    #remove escape chars and numbers
    data.iloc[i, 0] = re.sub(r'[\n\r]|[\d]', ' ', data.iloc[i, 0])
    # remove URLs
    data.iloc[i, 0] = re.sub(r'https?://\S+|www\.\S+', ' ', data.iloc[i, 0])


    # lemmatized_list = [lemmatizer.lemmatize(token) for token in data.iloc[i, 0].split(' ')]
    tokenized_string = word_tokenize(data.iloc[i, 0])
    lemmatized_list = [lemmatizer.lemmatize(token).lower() for token in tokenized_string]

    #use string not generator
    tokenized_list.append(' '.join(lemmatized_list))

# print(tokenized_list)

vectorizor = CountVectorizer()
cv_matrix = vectorizor.fit_transform(tokenized_list)

vectored_data = pd.DataFrame(data=cv_matrix.toarray(),columns = vectorizor.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(vectored_data, data.iloc[:,1], test_size = TEST_FRACTION, random_state = 0)

#NAIVE BAYES BEGIN
#getting probability for if SPAM

val_count = y_train.value_counts()
spam_length = val_count[1]
ham_length = val_count[0]
prior_prob_spam = val_count[1] / (val_count[0] + val_count[1])
prior_prob_ham = val_count[0] / (val_count[0] + val_count[1])
print('prior_prob_spam: ' + str(float(prior_prob_spam)))

spam_word_freq = {}
spamicity = {}
for col in X_train.columns:
    spam_word_freq[col] = 0

length = len(X_train)
print(X_train.shape)

for i, col in enumerate(X_train.columns):
    for index, value in X_train[col].items():
        # is spam
        if ((y_train[index] == 1) and (value > 0)):
            #col is the word
            spam_word_freq[col] += 1
    #smoothing
    spamicity[col] = (spam_word_freq[col]+1)/(spam_length+2)

    # print(i)
    # print(f"Number of spam emails with the word {col}: {spam_word_freq[col]}")
    # print(f"Spamicity of the word '{col}': {spamicity[col]} \n")

print('calced spamicity')
# print(spamicity)

ham_word_freq = {}
hamicity = {}
for col in X_train.columns:
    ham_word_freq[col] = 0

for i, col in enumerate(X_train.columns):
    for index, value in X_train[col].items():
        if ((y_train[index] == 0) and (value > 0)):
            ham_word_freq[col] += 1
    
    #smoothing
    hamicity[col] = (ham_word_freq[col]+1)/(ham_length+2)

print('calced hamicity')
# print(ham_word_freq)


correct = 0
cases = len(X_test)
for i in range(cases):
    row = pd.DataFrame(X_test.iloc[i]).transpose()
    org_index = row.index[0]

    PS = prior_prob_spam
    PH = prior_prob_ham
    PWordS = 0
    PWordH = 0

    for name, data in row.items():
        if (data.iloc[0] > 0):
            try:
                val = spamicity[name]
                # print(f'spam {name} {val}')
                val = val if (val != 0) else (1 / spam_length)
                PWordS += math.log(val)
            except KeyError:
                PWordS += math.log(1 / spam_length)
            
            try:
                val = hamicity[name]
                # print(f'ham {name} {val}')
                val = val if (val != 0) else (1 / ham_length)
                PWordH += math.log(val) 
            except KeyError:
                PWordH += math.log(1 / ham_length)
    
    PS = math.log(PS)
    PH = math.log(PH)
    #https://courses.cs.washington.edu/courses/cse312/18sp/lectures/naive-bayes/naivebayesnotes.pdf 
    isSpam = (PS + PWordS) > (PH + PWordH)
    prob = 1 if isSpam else 0

    print(f"~~~~~~~~~case {i}: {prob}    {y_test[org_index]}")
    if (round(prob) == y_test[org_index]):
        correct += 1

print(f'{correct} correct / {cases} cases')
print(f'{round(((correct / cases) * 100),ndigits=2)}% accuracy')


#BAD CAUSES UNDERFLOW
# correct = 0
# cases = len(X_test)
# for i in range(cases):
#     row = pd.DataFrame(X_test.iloc[i]).transpose()
#     org_index = row.index[0]

#     PS = prior_prob_spam
#     PH = prior_prob_ham
#     PWordS = 1
#     PWordH = 1

#     for name, data in row.items():
#         if (data.iloc[0] > 0):
#             # print('true 2')
#             try:
#                 # print('true 3')
#                 val = spamicity[name]
#                 PWordS *= val if (val != 0) else (1 / spam_length)
#             except KeyError:
#                 # print('true 4')
#                 print('needed smoothing spam?')
#                 PWordS *= 1 / spam_length
            
#             try:
#                 # print('true 5')
#                 val = hamicity[name]
#                 PWordH *= val if (val != 0) else (1 / ham_length)
#             except KeyError:
#                 # print('true 6')
#                 print('needed smoothing ham?')
#                 PWordH *= 1 / ham_length
    
#     prob = (PS * PWordS) / ((PS * PWordS) + (PH * PWordH))
#     print(PS)
#     print(PH)
#     print(PWordS)
#     print(PWordH)
#     print((PS * PWordS))
#     print(((PS * PWordS) + (PH * PWordH)))
#     print((PH * PWordH))
#     print(f"~~~~~~~~~case {i}: {prob}    {y_test[org_index]}")
#     if (round(prob) == y_test[org_index]):
#         correct += 1

# print(f'{correct} correct / {cases} cases')
# print(f'{round(((correct / cases) * 100),ndigits=2)}% accuracy')


# #V1
# #ML and plotting
# from sklearn import tree
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# #One hot encoding
# from sklearn.preprocessing import OneHotEncoder
# import matplotlib.pyplot as plt
# #import pandas
# import pandas as pd
# #to split the array
# import numpy as np

# #Auxillary
# import math
# from collections import OrderedDict

# #preprocessing
# import re
# import nltk
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# from nltk.tokenize import word_tokenize
# # nltk.download('punkt')
# # nltk.download('wordnet')

# data = pd.read_csv("spam.csv")
# data.drop(data.columns[[0, 1]], axis=1, inplace=True)
# data = data.iloc[:1000]

# tokenized_list = []

# # Removing punctuations in string
# # Using regex
# for i in range(len(data)):
#     #remove punctuation
#     data.iloc[i, 0] = re.sub(r'[^\w\s]', '', data.iloc[i, 0])
#     #remove escape chars and numbers
#     data.iloc[i, 0] = re.sub(r'[\n\r]|[\d]', ' ', data.iloc[i, 0])
#     # remove URLs
#     data.iloc[i, 0] = re.sub(r'https?://\S+|www\.\S+', ' ', data.iloc[i, 0])


#     # lemmatized_list = [lemmatizer.lemmatize(token) for token in data.iloc[i, 0].split(' ')]
#     tokenized_string = word_tokenize(data.iloc[i, 0])
#     lemmatized_list = [lemmatizer.lemmatize(token).lower() for token in tokenized_string]

#     #use string not generator
#     tokenized_list.append(' '.join(lemmatized_list))

# # print(tokenized_list)

# vectorizor = CountVectorizer()
# cv_matrix = vectorizor.fit_transform(tokenized_list)

# vectored_data = pd.DataFrame(data=cv_matrix.toarray(),columns = vectorizor.get_feature_names_out())

# X_train, X_test, y_train, y_test = train_test_split(vectored_data, data.iloc[:,1], test_size = 0.20, random_state = 0)

# #NAIVE BAYES BEGIN
# #https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/ 

# #getting probability for if SPAM

# val_count = y_train.value_counts()
# spam_length = val_count[1]
# ham_length = val_count[0]
# prior_prob_spam = val_count[1] / (val_count[0] + val_count[1])
# print('prior_prob_spam: ' + str(float(prior_prob_spam)))

# spam_word_freq = {}
# spamicity = {}
# # for col in X_train.columns:
# #     spam_word_freq[col] = 0

# length = len(X_train)
# print(X_train.shape)

# for i, col in enumerate(X_train.columns):
#     for index, value in X_train[col].items():
#         # is spam
#         if (y_train[index] == 1 and value > 0):
#             #col is the word
#             if (col in spam_word_freq):
#                 spam_word_freq[col] += 1
#             else:
#                 spam_word_freq[col] = 0
#     #smoothing
#     if (col in spam_word_freq):
#         spamicity[col] = (spam_word_freq[col]+1)/(spam_length+2)

#     # print(i)
#     # print(f"Number of spam emails with the word {col}: {spam_word_freq[col]}")
#     # print(f"Spamicity of the word '{col}': {spamicity[col]} \n")

# print('calced spamicity')
# print(spamicity)

# ham_word_freq = {}
# hamicity = {}
# # for col in X_train.columns:
# #     ham_word_freq[col] = 0

# for col in X_train.columns:
#     for index, value in X_train[col].items():
#         # is ham
#         if (y_train[index] == 0 and value > 0):
#             #col is the word
#             if (col in ham_word_freq):
#                 ham_word_freq[col] += 1
#             else:
#                 ham_word_freq[col] = 0
#     #smoothing
#     if (col in ham_word_freq):
#         hamicity[col] = (ham_word_freq[col]+1)/(ham+2)

# print('calced hamicity')

# naive_bayes = {}
# for col in X_train.columns:
#     #smoothing
#     PWordSpam = 0
#     try: 
#         PWordSpam = spamicity[col]
#     except KeyError:
#         PWordSpam = 1 / (spam_length + 2)

#     PSpam = prior_prob_spam

#     PWordHam = 0
#     try: 
#         PWordHam = hamicity[col]
#     except KeyError:
#         PWordHam = 1 / (ham_length + 2)

#     PHam = 1 - PSpam
#     # https://medium.com/@insight_imi/sms-spam-classification-using-na%C3%AFve-bayes-classifier-780368549279
#     naive_bayes[col] = (PWordSpam * PSpam) / ((PWordSpam * PSpam) + (PWordHam * PHam))

# print('calced naive bayes')
# print(naive_bayes)

# correct = 0
# cases = len(X_test)
# for i in range(cases):
#     row = pd.DataFrame(X_test.iloc[i, :]).transpose()
#     org_index = row.index[0]
#     prob = 1
#     for name, data in row.items():
#         if (data.iloc[0] > 0):
#             try:
#                 print(f'{name} {naive_bayes[name]}')
#                 val = naive_bayes[name]
#                 prob *= val
#                 # prob *= (val != 0) if val else 1/(spam_length+2)
#             except KeyError:
#                 print(f'smooth {1/(length+2)}')
#                 prob *= 1/(length+2)
#             print(f'prob now is: {prob}')
        

#     print(f"~~~~~~~~~case {i}: {prob}    {y_test[org_index]}")
#     if (round(prob) == y_test[org_index]):
#         correct += 1

# print(f'{correct} correct / {cases} cases')
# print(f'{round(((correct / cases) * 100),ndigits=2)}% accuracy')



# try:
#             pr_WS = spamicity[word]
#         except KeyError:
#             pr_WS = 1/(total_spam+2)  # Apply smoothing for word not seen in spam training data, but seen in ham training 
#             print(f"prob '{word}' is a spam word: {pr_WS}")




# print(data.head())

# print(data.describe(include = 'all'))

# print(data.groupby('label_num').describe())
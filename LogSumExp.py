import numpy as np
import math

def stable_log_sum_exp(log_terms):
    """
    Compute log(sum(exp(log_terms))) in a numerically stable way.
    """
    M = max(log_terms)  # Find max log term for stability
    return M + np.log(np.sum(np.exp(np.array(log_terms) - M)))

# # Given numerator [probability A, [probability x | A]]
# # Given denominator list [[probability A, [probability x | A]]]
def sum_log_exp_naive_bayes(num: list[float, list[float]], denom: list[list[float, list[float]]]) -> float:
    """
    Compute log( P(A) * prod(P(x | A)) / sum(P(A_i) * prod(P(x | A_i))) ) 
    in a numerically stable way using the log-sum-exp trick.

    Parameters:
    - num: A list where the first element is P(A) and the second element is a list of conditional probabilities P(x | A).
    - denom: A list of lists where each sublist is structured like num, representing different cases of P(A_i) and P(x | A_i).

    Returns:
    - log probability of the numerator divided by the denominator
    """

    # Compute log of numerator: log(P(A)) + sum(log(P(x | A)))
    log_num = np.log(num[0]) + np.sum(np.log(num[1]))
    # print (f'num[0]{num[0]}n  p.log(num[0]) {np.log(num[0])} np.sum(np.log(num[1])) {np.sum(np.log(num[1]))}')

    # Compute log of denominator for each term in denom
    log_denom_terms = [np.log(d[0]) + np.sum(np.log(d[1])) for d in denom]

    # Compute log-sum-exp for denominator
    log_denom = stable_log_sum_exp(log_denom_terms)

    # Compute final result: log(numerator) - log(denominator)

    print (f'log_num {round(log_num,6)} log_denom {round(log_denom, 6)} exp{round(math.exp(log_num - log_denom), 6)}')
    return math.exp(log_num - log_denom)


# # Example usage:
# num = (0.3, [0.8, 0.7, 0.9])  # P(A) and P(x | A)
# denom = [
#     (0.3, [0.8, 0.7, 0.9]),  # P(A1) and P(x | A1)
#     (0.2, [0.6, 0.5, 0.7]),  # P(A2) and P(x | A2)
#     (0.5, [0.9, 0.8, 0.95])  # P(A3) and P(x | A3)
# ]

# result = sum_log_exp_naive_bayes(num, denom)
# print(result)










# # def stable_log_sum_exp_multiple(terms):
# #     """
# #     Compute log(a_1 * prod(x_1) + a_2 * prod(x_2) + ... + a_n * prod(x_n)) 
# #     in a numerically stable way using the log-sum-exp trick.

# #     terms: List of tuples (a_i, x_list_i), where:
# #         - a_i is the constant for the i-th term.
# #         - x_list_i is the list of variables for the i-th term.
# #     """
# #     log_terms = []
    
# #     # Compute log(a * prod(x)) for each term
# #     for term in terms:
# #         # print("term")
# #         # print(term)
# #         a = term[0]
# #         x_list = term[1]
# #         log_term = np.log(a) + np.sum(np.log(x_list))  # log(a) + sum(log(x))
# #         log_terms.append(log_term)
    
# #     # Find the maximum value for numerical stability
# #     M = max(log_terms)
    
# #     # Apply log-sum-exp trick
# #     result = M + np.log(np.sum(np.exp(np.array(log_terms) - M)))
    
# #     return result


# # Given numerator [probability A, [probability x | A]]
# # Given denominator list [[probability A, [probability x | A]]]
# def sum_log_exp_naive_bayes(num: list[float, list[float]], denom: list[list[float, list[float]]]) -> float:
#     # Numerator
#     numerator = 0
#     prob_A_num = num[0]
#     prob_A_given_X_list = num[1]

#     numerator += math.log(prob_A_num)
#     for prob in prob_A_given_X_list:
#         numerator += math.log(prob)
    
#     # Denominator
#     denominator = 0

#     max_term = 0
#     for term in denom:
#         val = 0
#         prob_A = term[0]    
#         val += math.log(prob_A)
        
#         prob_A_given_X_list = term[1]
#         for prob_cond in prob_A_given_X_list:
#             val += math.log(prob_cond)

#         if (val > max_term):
#             max_term = val

#     denominator += max_term
#     denom_summer = 0
    
#     for term in denom:
#         val = 0
#         prob_A = term[0]
#         val += math.log(prob_A)
        
#         prob_A_given_X_list = term[1]
#         for prob_cond in prob_A_given_X_list:
#             val += math.log(prob_cond)

#         val -= max_term
#         denom_summer += math.exp(val)
#     denominator += math.log(denom_summer)

#     probability = math.exp(numerator - denominator)
        
    
#     # prob_list_denom = denom[i]
#     # prob_A_denom = prob_list_denom[0]
#     # prob_A_given_X_list = prob_list_denom[1]
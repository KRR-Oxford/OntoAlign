import numpy as np


def cal_result_score(Y, scores, alpha, n):
    pos, true_pos, true_neg, false_neg = 0, 0, 0, 0
    for i in range(n):
        score = scores[i]
        y = Y[i]
        if score >= alpha:
            pos += 1
            if y == 1:
                true_pos += 1
        if score < alpha and y == 0:
            true_neg += 1
        if score < alpha and y == 1:
            false_neg += 1
    precision = true_pos / pos if pos > 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    acc = (true_pos + true_neg) / n
    return precision, recall, f1, acc


'''
def cal_result_dist(Y, dist, alpha, n):
    pos, true_pos, true_neg, false_neg = 0, 0, 0, 0
    for i in range(n):
        y = Y[i]
        if dist[i] <= alpha:
            pos += 1
            if y == 1:
                true_pos += 1
        if dist[i] > alpha and y == 0:
            true_neg += 1
        if dist[i] > alpha and y == 1:
            false_neg += 1
    precision = true_pos / pos if pos > 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    acc = (true_pos + true_neg) / n
    return precision, recall, f1, acc
'''


def threshold_searching(Y, scores, num):
    max_valid_f1, max_valid_p, max_valid_r, max_valid_acc, max_alpha = -np.inf, -np.inf, -np.inf, -np.inf, 0
    for alpha in np.arange(0, 1, 0.02):
        valid_p, valid_r, valid_f1, valid_acc = cal_result_score(Y=Y, scores=scores, alpha=alpha, n=num)
        print('alpha: %.2f, precision: %.3f, recall: %.3f, f1 score: %.3f, accuracy: %.3f'
              % (alpha, valid_p, valid_r, valid_f1, valid_acc))
        if valid_f1 > max_valid_f1:
            max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc = alpha, valid_f1, valid_p, valid_r, valid_acc
    return max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc

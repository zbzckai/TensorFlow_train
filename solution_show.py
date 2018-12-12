# -*- coding: utf-8 -*-
# @Time    : 2018\11\19 0019 10:19
# @Author  : 凯
# @File    : solution_show.py

import pandas as pd

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.externals import joblib
os.getcwd()
os.chdir('E:\work\\20181102train')
solution = joblib.load('solution_intermediate_41000.m')



columns_name = ['test_name', 'true_label'
    , 'pred_1_lable', 'pred_1_p'
    , 'pred_2_lable', 'pred_2_p'
    , 'pred_3_lable', 'pred_3_p'
    , 'pred_4_lable', 'pred_4_p'
    , 'pred_5_lable', 'pred_5_p']
test_solution = pd.DataFrame(solution, columns=columns_name)

def replace_fun(str_name):
    return str_name.replace('_', ' ')


test_solution['top_5soc'] = test_solution['pred_1_p'] + test_solution['pred_2_p'] \
                            + test_solution['pred_3_p'] + test_solution['pred_4_p'] + \
                            test_solution['pred_5_p']
test_solution = test_solution.sort_values(by='top_5soc').reset_index(drop=True)
test_solution['true_label'] = test_solution['true_label'].apply(lambda x: replace_fun(x))

#def is_in_top5(series_str):
#    return ((series_str[ 'true_label'].tolist()[0] == series_str['pred_1_lable'].values.tolist()[0]) |
#    (series_str[ 'true_label'].tolist()[0] == series_str['pred_2_lable'].values.tolist()[0]) |
#    (series_str[ 'true_label'].tolist()[0] == series_str['pred_3_lable'].values.tolist()[0]) |
#    (series_str[ 'true_label'].tolist()[0] == series_str['pred_4_lable'].values.tolist()[0]) |
#    (series_str[ 'true_label'].tolist()[0] == series_str['pred_5_lable'].values.tolist()[0]) )
#
#            in (series_str[['pred_1_lable'
#        , 'pred_2_lable'
#        , 'pred_3_lable'
#        , 'pred_4_lable'
#        , 'pred_5_lable']].values.tolist()[0]))
#
#def is_in_top5(series_str):
#    return (series_str.loc[series_str.index, 'true_label'].tolist()[0] ==
#            in (series_str[['pred_1_lable'
#        , 'pred_2_lable'
#        , 'pred_3_lable'
#        , 'pred_4_lable'
#        , 'pred_5_lable']].values.tolist()[0]))



test_solution['is_true_1'] = test_solution['true_label'] == test_solution['pred_1_lable']
test_solution['is_true_2'] = test_solution['true_label'] == test_solution['pred_2_lable']
test_solution['is_true_3'] = test_solution['true_label'] == test_solution['pred_3_lable']
test_solution['is_true_4'] = test_solution['true_label'] == test_solution['pred_4_lable']
test_solution['is_true_5'] = test_solution['true_label'] == test_solution['pred_5_lable']
test_solution['is_in_top5'] = (test_solution['is_true_1'] | test_solution['is_true_2'] |
test_solution['is_true_3'] |
test_solution['is_true_4'] |
test_solution['is_true_5'] )
del test_solution['is_true_2']
del test_solution['is_true_3']
del test_solution['is_true_4']
del test_solution['is_true_5']
#test_solution['is_in_top5'] = test_solution.groupby(by='test_name', as_index=False).apply(lambda x: is_in_top5(x))
sum(test_solution['is_true_1']) / test_solution.shape[0]
sum(test_solution['is_in_top5']) / test_solution.shape[0]

test_solution.to_csv('solution_output_graph.csv',encoding='gbk')
print('准确率')
print(sum(test_solution['is_true_1']) / test_solution.shape[0])
print('是否在top_5')
print(sum(test_solution['is_in_top5']) / test_solution.shape[0])
"""
"""for i, j in test_solution.groupby(by='test_name', as_index=False):
    ceshi1 = i
    ceshi2 = j
    if ceshi2.shape[0] != 1:
        print(ceshi2)
"""
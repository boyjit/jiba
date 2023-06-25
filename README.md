import pandas as pd
from causalinference import CausalModel
from sklearn.linear_model import LinearRegression

# Causal Learning Dataset citation
# Dataset: A Semi-synthetic Dataset For Causal Learning and Decision Optimization
# Author: Bochen Lv
# Retrieved from: https://github.com/DataCanvasIO/WAIC-2022-Hackathon-Causal-Learning-and-Decision-Optimization-Challenge
# Year: 2022

# 加载训练集和测试集数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 提取特征列、干预方案列和结果列
features = train_data[['V_0', 'V_1']]
treatment = train_data['treatment']
outcome = train_data['outcome']

# 构建因果模型
causal_model = CausalModel(
    Y=outcome.values,
    D=treatment.values,
    X=features.values
)

# 估计因果效应
causal_model.est_via_ols(adj=1)
effect_1 = causal_model.estimates['ate']
causal_model.est_via_ols(subset=(treatment == 2), adj=1)
effect_2 = causal_model.estimates['ate']

# 使用训练集估计得到的因果效应对测试集进行预测
test_features = test_data[['V_0', 'V_1']]
predicted_outcome_1 = LinearRegression().fit(features, outcome).predict(test_features) + effect_1
predicted_outcome_2 = LinearRegression().fit(features, outcome).predict(test_features) + effect_2

# 创建结果DataFrame并保存为CSV文件
result_df = pd.DataFrame({'outcome_1': predicted_outcome_1, 'outcome_2': predicted_outcome_2})
result_df.to_csv('result.csv', index=False)

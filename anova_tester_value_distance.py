
import pandas as pd
import random
import numpy as np

# read original dataset
file1 = pd.read_csv('memory_time/AV_outdata_s_v_g_e.csv')
file2 = pd.read_csv('memory_time/AV_outdata_s_v_g_ne.csv')
file3 = pd.read_csv('memory_time/AV_outdata_s_v_g_int.csv')

# we look at the minimum distance between the two vehicle

min_distance_ee = file1.groupby('it')['distance'].min().reset_index()
min_distance_ne = file2.groupby('it')['distance'].min().reset_index()
min_distance_int = file3.groupby('it')['distance'].min().reset_index()

# min_distance_ee = file1.groupby('it')['distance'].sum().reset_index()
# min_distance_ne = file2.groupby('it')['distance'].sum().reset_index()
# min_distance_int = file3.groupby('it')['distance'].sum().reset_index()

#we look at sum of the reward for each iteration and then we look at the mean for all of them
sum_reward_it_ee = file1.groupby('it')['Value'].sum().reset_index()
sum_reward_it_ne = file2.groupby('it')['Value'].sum().reset_index()
sum_reward_it_int = file3.groupby('it')['Value'].sum().reset_index()

# sum_reward_it_ee = file1.groupby('it')['Value'].max().reset_index()
# sum_reward_it_ne = file2.groupby('it')['Value'].max().reset_index()
# sum_reward_it_int = file3.groupby('it')['Value'].max().reset_index()

# sum_reward_it_ee = file1.groupby('it')['Value'].mean().reset_index()
# sum_reward_it_ne = file2.groupby('it')['Value'].mean().reset_index()
# sum_reward_it_int = file3.groupby('it')['Value'].mean().reset_index()

# mean_reward_it_ee = file1.groupby('it')['Value'].mean().reset_index()
# mean_reward_it_ne = file2.groupby('it')['Value'].mean().reset_index()
# mean_reward_it_int = file3.groupby('it')['Value'].mean().reset_index()


#
import statistics
#mean and standard deviation
std_reward_ee = statistics.stdev(sum_reward_it_ee['Value'])
std_reward_ne = statistics.stdev(sum_reward_it_ne['Value'])
std_reward_int = statistics.stdev(sum_reward_it_int['Value'])

average_ee = statistics.mean(sum_reward_it_ee['Value'])
average_ne = statistics.mean(sum_reward_it_ne['Value'])
average_int = statistics.mean(sum_reward_it_int['Value'])

std_distance_ee = statistics.stdev(min_distance_ee['distance'])
std_distance_ne = statistics.stdev(min_distance_ne['distance'])
std_distance_int = statistics.stdev(min_distance_int['distance'])

average_dist_ee = statistics.mean(min_distance_ee['distance'])
average_dist_ne = statistics.mean(min_distance_ne['distance'])
average_dist_int = statistics.mean(min_distance_int['distance'])



df = pd.DataFrame([['Empathetic', average_ee, std_reward_ee, average_dist_ee, std_distance_ee],
                   ['Non-empathetic', average_ne, std_reward_ne, average_dist_ne, std_distance_ne],
                   ['Intermittent', average_int, std_reward_int, average_dist_int, std_distance_int]],
                  columns = ['Case', 'Reward (mean)', 'Reward Std', 'Min distance','Std distance'])



#adding it in one file
print(df)
sample_df_int =pd.DataFrame({"Value": sum_reward_it_int['Value'],
                         "Distance": min_distance_int['distance'], "Inference": "Int"})

#print(sample_df_int)


sample_df_ee =pd.DataFrame({"Value": sum_reward_it_ee['Value'],
                         "Distance": min_distance_ee['distance'], "Inference": "Emp"})

#print(sample_df_ee)

sample_df_ne =pd.DataFrame({"Value": sum_reward_it_ne['Value'],
                         "Distance": min_distance_ne['distance'], "Inference": "NE"})

#print(sample_df_ne)
sample_df_int = sample_df_int.append(sample_df_ee, ignore_index=True)
#sample_df_int = sample_df_int.append(sample_df_ne, ignore_index=True)
print(len(sample_df_int))
print(sample_df_int)
sample_df = sample_df_int


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
#
unique_majors = sample_df['Inference'].unique()
for major in unique_majors:
    stats.probplot(sample_df[sample_df['Inference'] == major]['Value'], dist="norm", plot=plt)
    plt.title("Probability Plot - " +  major)
    plt.show()

# calculate ratio of the largest to the smallest sample standard deviation
ratio = sample_df.groupby('Inference').std().max() / sample_df.groupby('Inference').std().min()


#
data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']]
anova_table = pd.DataFrame(data, columns = ['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit'])
anova_table.set_index('Source of Variation', inplace = True)
#
# # calculate SSTR and update anova table
x_bar = sample_df['Value'].mean()
SSTR = sample_df.groupby('Inference').count() * (sample_df.groupby('Inference').mean() - x_bar)**2
anova_table['SS']['Between Groups'] = SSTR['Distance'].sum()
#
# # calculate SSE and update anova table
SSE = (sample_df.groupby('Inference').count() - 1) * sample_df.groupby('Inference').std()**2
anova_table['SS']['Within Groups'] = SSE['Value'].sum()
#
# # calculate SSTR and update anova table
SSTR = SSTR['Value'].sum() + SSE['Value'].sum()
anova_table['SS']['Total'] = SSTR

# # update degree of freedom
anova_table['df']['Between Groups'] = sample_df['Inference'].nunique() - 1
anova_table['df']['Within Groups'] = sample_df.shape[0] - sample_df['Inference'].nunique()
anova_table['df']['Total'] = sample_df.shape[0] - 1
#
# calculate MS
anova_table['MS'] = anova_table['SS'] / anova_table['df']

# calculate F
F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
anova_table['F']['Between Groups'] = F
#
# p-value
anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

# F critical
alpha = 0.05
# possible types "right-tailed, left-tailed, two-tailed"
tail_hypothesis_type = "two-tailed"
if tail_hypothesis_type == "two-tailed":
    alpha /= 2
anova_table['F crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

# Final ANOVA Table
print(anova_table)

# The p-value approach
print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
conclusion = "Failed to reject the null hypothesis."
if anova_table['P-value']['Between Groups'] <= alpha:
    conclusion = "Null Hypothesis is rejected."
print("F-score is:", anova_table['F']['Between Groups'], " and p value is:", anova_table['P-value']['Between Groups'])
print(conclusion)

# The critical value approach
print("\n--------------------------------------------------------------------------------------")
print("Approach 2: The critical value approach to hypothesis testing in the decision rule")
conclusion = "Failed to reject the null hypothesis."
if anova_table['F']['Between Groups'] > anova_table['F crit']['Between Groups']:
    conclusion = "Null Hypothesis is rejected."
print("F-score is:", anova_table['F']['Between Groups'], " and critical value is:",
      anova_table['F crit']['Between Groups'])
print(conclusion)
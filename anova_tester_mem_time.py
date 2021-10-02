
import pandas as pd
import random
import numpy as np

# read original dataset
student_df = pd.read_csv('students.csv')

file1 = open("memory_time/test_memory_log_ee.txt")
file2 = open("memory_time/train_memory_log_ne.txt")
file3 = open("memory_time/test_memory_log_int.txt")
data1 = file1.read()
data1 = data1.split("\n")
stuff1 = []

data2 = file2.read()
data2 = data2.split("\n")
stuff2 = []

data3 = file3.read()
data3 = data3.split("\n")
stuff3 = []

for each in data1:
    stuff1.append(each.split(',')[0])

for each in data2:
    stuff2.append(each.split(',')[0])

for each in data3:
    stuff3.append(each.split(',')[0])

memory_info_ee = []
for each in stuff1:
    if '110' in each:
        inf = each.split('   ')
        memory_info_ee.append(float(inf[2][:-4]))

memory_info_ne = []
for each in stuff2:
    if '110' in each:
        inf = each.split('   ')
        memory_info_ne.append(float(inf[2][:-4]))

memory_info_int = []
for each in stuff3:
    if '107' in each:
        inf = each.split('   ')
        memory_info_int.append(float(inf[2][:-4]))



print(len(memory_info_ee))
print(len(memory_info_ne))
print(len(memory_info_int))
#memory calc
average_ee = np.sum(memory_info_ee)/len(memory_info_ee)
average_ne = np.sum(memory_info_ne)/len(memory_info_ne)
average_int = np.sum(memory_info_int)/len(memory_info_int)

#time calc
file1 = open("memory_time/test_time_log_ee.txt")
data1 = file1.read()
data1 = data1.split("\n")

times_ee = []
for each in data1:
    try:
        times_ee.append(np.float(each.split(" ")[3]))
    except:
        pass

average_time_ee = np.sum(times_ee)/len(times_ee)
max_time_ee = np.max(times_ee)
max_idx_ee = np.argmax(times_ee)


print("Average time:", average_time_ee, "seconds")
print("Max time:", max_time_ee, "seconds at episode", max_idx_ee)
print("Total episodes:", len(times_ee))

file2 = open("memory_time/train_time_log_ne.txt")
data2 = file2.read()
data2 = data2.split("\n")

times_ne = []
for each in data2:
    try:
        times_ne.append(np.float(each.split(" ")[3]))
    except:
        pass

average_time_ne = np.sum(times_ne)/len(times_ne)
max_time_ne = np.max(times_ne)
max_idx_ne = np.argmax(times_ne)

print("Average time:", average_time_ne, "seconds")
print("Max time:", max_time_ne, "seconds at episode", max_idx_ne)
print("Total episodes:", len(times_ne))

file3 = open("memory_time/test_time_log_int.txt")
data3 = file3.read()
data3 = data3.split("\n")

times_int = []
for each in data3:
    try:
        times_int.append(np.float(each.split(" ")[3]))
    except:
        pass

average_time_int = np.sum(times_int)/len(times_int)
max_time_int = np.max(times_int)
max_idx_int = np.argmax(times_int)

print("Average time:", average_time_int, "seconds")
print("Max time:", max_time_int, "seconds at episode", max_idx_int)
print("Total episodes:", len(times_int))

df_tester = pd.DataFrame([times_int, "ent"])

import statistics
std_time_ee = statistics.stdev(times_ee)
std_time_ne = statistics.stdev(times_ne)
std_time_int = statistics.stdev(times_int)

std_mem_ee = statistics.stdev(memory_info_ee)
std_mem_ne = statistics.stdev(memory_info_ne)
std_mem_int = statistics.stdev(memory_info_int)
print(std_time_ee, std_time_ne, std_time_int, std_mem_ee, std_mem_ne, std_mem_int )

df = pd.DataFrame([['Empathetic', average_ee, average_time_ee], ['Non-empathetic', average_ne, average_time_ne],
                   ['Intermittent', average_int, average_time_int]],
                  columns = ['Case', 'Average Memory (MB)', 'Average Time (seconds)'])


#adding it in one file
print(df)
sample_df_int =pd.DataFrame({"Memory": memory_info_int[-len(times_int):],
                         "Time": times_int, "Inference": "Int"})

#print(sample_df_int)


sample_df_ee =pd.DataFrame({"Memory": memory_info_ee[-len(times_ee):],
                         "Time": times_ee, "Inference": "Emp"})

#print(sample_df_ee)

sample_df_ne =pd.DataFrame({"Memory": memory_info_ne[-len(times_ne):],
                         "Time": times_ne, "Inference": "NE"})

#print(sample_df_ne)
#sample_df_int = sample_df_int.append(sample_df_ee, ignore_index=True)
sample_df_int = sample_df_int.append(sample_df_ne, ignore_index=True)
#sample_df_ee = sample_df_ee.append(sample_df_ne, ignore_index=True)
print(len(sample_df_int))
print(sample_df_int)
sample_df = sample_df_int
#sample_df = sample_df_ee


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

unique_majors = sample_df['Inference'].unique()
for major in unique_majors:
    stats.probplot(sample_df[sample_df['Inference'] == major]['Time'], dist="norm", plot=plt)
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
x_bar = sample_df['Time'].mean()
SSTR = sample_df.groupby('Inference').count() * (sample_df.groupby('Inference').mean() - x_bar)**2
anova_table['SS']['Between Groups'] = SSTR['Time'].sum()
#
# # calculate SSE and update anova table
SSE = (sample_df.groupby('Inference').count() - 1) * sample_df.groupby('Inference').std()**2
anova_table['SS']['Within Groups'] = SSE['Time'].sum()
#
# # calculate SSTR and update anova table
SSTR = SSTR['Time'].sum() + SSE['Time'].sum()
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
import numpy as np
#mu_p, sigma_p = 0, 0.0125 # mean and standard deviation
mu_p, sigma_p = 0, 0.1
s = np.random.normal(mu_p, sigma_p, 100*200)
print(s)

mu_v, sigma_v = 0, 0.005
v = np.random.normal(mu_v, sigma_v, 100)
print(min(v), max(v), np.mean(v), np.std(v))
print(min(s), max(s), np.mean(s), np.std(s))


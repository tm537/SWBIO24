# SWBIO24
 
 # import data 

import pandas as pd

clone = ("./broth_clone.csv")
clone_data = pd.read_csv(clone)

# Visualise data

import seaborn as sns

sns.catplot(
    data=clone_data,
    x="Env",
    y="combined.density",
    kind="box",
)

# Transform data to help visualisation

import numpy as np

clone_data['combined_density_lg'] = np.log10(clone_data['combined.density'])

clone_data['summed_mono_lg'] = np.log10(clone_data['summed_mono'])

clone_data

# Plot data and check that covariate has linear reationship with response variable 

sns.relplot(
    data=clone_data,
    x="summed_mono_lg",
    y="combined_density_lg",
    hue="Env",
    style="Env",
).set(
    xlabel="Summed mono",
    ylabel="Coculture",
)

# this plot shows us that we are meeting an assumption and that this is an interesting relationship to investigate further

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# fit ANCOVA using OLS and specifying Env as categorical variable and summed_mono as covriate

model2 = ols('combined_density_lg ~ C(Env)+ summed_mono_lg', data=clone_data).fit()

print(model2.summary()) 

# Print output of 

# Plotting residuals to check model 
import matplotlib.pyplot as plt

residuals = model2.resid
fig, ax = plt.subplots()
ax.scatter(model2.predict(), residuals)
ax.axhline(0, color='red', lw=2)
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs. Predicted')
plt.show()

# Normality test for residuals is ok here as n is under 50 
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print('Shapiro-Wilk Test: p-value=%.3f' % (p))

# INTALL THIS PACKAGE
pip install pingouin 

# pimgouin to check P value of Env and see if it signficantly affects summed coculture
# and the effect of the covariate 

from pingouin import ancova, read_dataset


ancova(data=clone_data, dv='combined_density_lg', covar='summed_mono_lg', between='Env')

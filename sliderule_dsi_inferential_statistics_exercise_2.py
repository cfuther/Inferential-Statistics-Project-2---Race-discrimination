
# coding: utf-8

# # Examining Racial Discrimination in the US Job Market
# 
# ### Background
# Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.
# 
# ### Data
# In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.
# 
# Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

# <div class="span5 alert alert-info">
# ### Exercises
# You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.
# 
# Answer the following questions **in this notebook below and submit to your Github account**. 
# 
#    1. What test is appropriate for this problem? Does CLT apply?
#    2. What are the null and alternate hypotheses?
#    3. Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches.
#    4. Write a story describing the statistical significance in the context or the original problem.
#    5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?
# 
# You can include written notes in notebook cells using Markdown: 
#    - In the control panel at the top, choose Cell > Cell Type > Markdown
#    - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# 
# 
# #### Resources
# + Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
# + Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# + Formulas for the Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
# </div>
# ****

# In[2]:


import pandas as pd
import numpy as np
from scipy import stats


# In[4]:


data = pd.io.stata.read_stata('C:/DS/Springboard/EDA_racial_discrimination/data/us_job_market_discrimination.dta')


# In[5]:


# number of callbacks for black-sounding names
sum(data[data.race=='w'].call)


# In[7]:


sum(data[data.race=='b'].call)


# In[6]:


data.head()


# <div class="span5 alert alert-success">
# <p>Your answers to Q1 and Q2 here</p>
# </div>
# # Answer to Q1:
# Since the sample size is relatively large (n=392), z-test will be appropriate for this case. The sampling process is random and indepedent. Say p represents probability of call to white-sounding applicant, n*p and n*(1-p) are both greater than 10. Therefore, Central limit theorem applies. 
# # Answer to Q2:
# Null hypothesis: race does not have an impact on the rate of callbacks for resumes. In other words, mean callback rates of white-sounding and black-sounding names are the same
# 
# Alternative hypothesis: mean callback rates of white-sounding and black-sounding names are different

# In[8]:


w = data[data.race=='w']
b = data[data.race=='b']


# In[19]:


print(len(w))
print(len(b))
print(np.sum(w.call))
print(np.sum(b.call))


# In[31]:


# Your solution to Q3 here
# two sample z-test
mar_er =  np.sqrt(w.call.var()/len(w)+b.call.var()/len(b))  
dif_obs = w.call.mean() - b.call.mean()
print("margin of error equals ",mar_er)
con_hi = dif_obs + 1.96*mar_er
con_lw = dif_obs - 1.96*mar_er
print("confidence interval at 95% is ",con_lw,"~",con_hi)
z = dif_obs/mar_er
p = 0 #approximately 0
print("with z value as",z,", p-value equals",p)


# In[32]:


# Your solution to Q3 here
# bootstrap

rep_bs = np.empty(10000)
for i in range(10000):
    bs_w = np.random.choice(w.call,len(w))
    bs_b = np.random.choice(b.call,len(b))
    rep_bs[i] = np.sum(bs_w)/len(bs_w) - np.sum(bs_b)/len(bs_b)
p_bs = np.sum(rep_bs < 0)/10000
print("p-value is",p_bs,"based on bootstrapping.")


# <div class="span5 alert alert-success">
# <p> Your answers to Q4 and Q5 here </p>
# </div>
# # Answer to Q4:
# As p-value from z-test and bootstrapping is approximately 0, less than significance level 5%. Therefore,the null hypothesis can be rejected and callback rates of white and black-sounding applicants are different
# # Answer to Q5:
# No, results from previous analysis mean the callback decision is related to race. There maybe other factors having stronger influence on the decision than race. 
# 
# In order to identify the influencing power of race, one possible approach is to use logistic regression. Slope represents how the callback decision changes with race given all other variables remain the same. If the slope of race is smaller than the slopes of other parameters, it means the parameters with greater slopes have higher influence on the decision.

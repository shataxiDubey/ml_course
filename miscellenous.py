import pandas as pd
import numpy as np
P = 5
N = 30
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})

cond  = X.dtypes == 'category'

print(X.dtypes == 'category')
if cond.all():
    print('Categorical datatype')
else:
    print('non categorical datatype')

X = X.drop([0], axis =0)
print(X)
print(X.dtypes)
import pandas as pd

num_patients = 1000
df = pd.read_csv(f"../Data Generation/sphn_transductive_{num_patients}_0.nt", sep=" ", header=None)
df.drop(columns=df.columns[-1], axis=1, inplace=True)
df.columns=['s', 'r', 'd']

df['r'].unique()
time_df = df[df['r'].str.contains('hasStartDateTime|hasDeterminationDateTime')]
duplicates = time_df[time_df.duplicated(subset=['d'], keep=False)].sort_values(by=['d'])

for i, v in duplicates['s'].items():
    temp = df[df['s'] == v]
    duplicates = pd.concat((duplicates, temp), axis=0)
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('test_res.csv', index_col=0)
group = df[['class', 'rmse', 'pgp']].groupby('class')
df_group = group.mean().sort_values('rmse')

fig, ax = plt.subplots()
for i, row in enumerate(df_group.iterrows()):
    ax.plot(alphas, row[1]['pgp'], 'o-',
            lw=2,
            mec='k',
            label=row[0].replace('_', ' '))
ax.set(xlabel=r'$\alpha$', ylabel=r'PGP($\alpha$)')
ax.legend()
sns.despine()
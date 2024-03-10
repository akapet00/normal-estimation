import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def update_rc():
    sns.set(style='ticks',
            rc={'lines.linewidth': 2.5,
                'lines.markersize': 7.5,
                'lines.markeredgecolor': 'k',
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{lmodern}',
                'font.family': 'lmodern',
                'font.size': 11,
                'axes.titlesize': 11,
                'axes.labelsize': 11,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.titlesize': 11})


def load_data(fname):
    df = pd.read_csv(fname, index_col=0)
    df['pgp'] = df['pgp'].apply(
        lambda x: np.fromstring(x.strip().replace('[', '').replace(']', ''),
                                sep=' ')
    )
    cols = ['class', 'rmse', 'pgp']
    group = df[cols].groupby('class')
    return group.mean().sort_values('rmse')


def main():
    # create dataset
    fname = 'test_res.csv'
    df = load_data(fname)
    
    # adjust matplotlib parameters
    update_rc()

    # plotting settings
    alphas = [0, 5, 10, 15, 20, 25, 30]
    markers = ['o', 'X', '^', 's', 'd', 'h']
    colors = sns.color_palette('husl', 6)
    width = 5.8

    # visualization
    fig = plt.figure(figsize=(width, width/(4/3)))
    ax = plt.axes()
    for i, row in enumerate(df.iterrows()):
        ax.plot(alphas, row[1]['pgp'], '-',
                color=colors[i],
                marker=markers[i],
                label=row[0].replace('_', ' '))
    ax.set(xlabel=r'$\alpha$', ylabel=r'PGP($\alpha$)')
    ax.legend()
    sns.despine()
    fig.tight_layout()
    plt.show()
    fig.savefig('pgp.pdf', bbox_inches='tight')

    
if __name__ == '__main__':
    main()

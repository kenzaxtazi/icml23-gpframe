
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Make the font bigger
plt.rcParams.update({'font.size': 22})


def plot():
    """
    Comparison plot with:
        a = training data
        b = k-NN
        c = SE-ARD GP
        d = original implementation
        e = framework sparse GP
        f = framework chunks
    """

    # Load training data
    df_train = pd.read_csv('data/plot_train_data.csv')

    # Load original implementation
    df_og = pd.read_csv('data/og_output.csv')

    # Load knn benchmark
    df_knn = pd.read_csv('data/knn_output.csv')

    # Load SE kernel with ARD benchmark
    df_se_ard = pd.read_csv('data/se_ard_output.csv')

    # Load framework output
    df_frame = pd.read_csv('data/frame_chunk_output.csv')

    # Load framework sparse GP output
    df_sparse = pd.read_csv('data/frame_output.csv')

    # Plot

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

    # ROW 1
    pcm = ax[0, 0].scatter(df_train['x']/1000, df_train['y']/1000, c=df_train['dhdt_filt'],
                           vmin=df_train['dhdt_filt'].min(), vmax=df_train['dhdt_filt'].min()*-1, cmap='coolwarm')
    ax[0, 0].set_ylabel('y [km]')
    ax[0, 0].text(x=5, y=-997, s='a')

    ax[0, 1].scatter(df_frame['x']/1000, df_frame['y']/1000, c=df_knn['ypred'], cmap='coolwarm',
                     vmin=df_train['dhdt_filt'].min(), vmax=df_train['dhdt_filt'].min()*-1)
    ax[0, 1].text(x=5, y=-997, s='b')

    ax[0, 2].scatter(df_se_ard['x']/1000, df_se_ard['y']/1000, c=df_se_ard['ypred'], cmap='coolwarm',
                     vmin=df_train['dhdt_filt'].min(), vmax=df_train['dhdt_filt'].min()*-1)
    ax[0, 2].text(x=5, y=-997, s='c')

    # ROW 2

    ax[1, 0].scatter(df_og['x']/1000, df_og['y']/1000, c=df_og['ypred'], cmap='coolwarm',
                     vmin=df_train['dhdt_filt'].min(), vmax=df_train['dhdt_filt'].min()*-1)
    ax[1, 0].set_ylabel('y [km]')
    ax[1, 0].set_xlabel('x [km]')
    ax[1, 0].text(x=5, y=-997, s='d')

    ax[1, 1].scatter(df_sparse['x']/1000, df_sparse['y']/1000, c=df_sparse['ypred'], cmap='coolwarm',
                     vmin=df_train['dhdt_filt'].min(), vmax=df_train['dhdt_filt'].min()*-1)
    ax[1, 1].set_xlabel('x [km]')
    ax[1, 1].text(x=5, y=-997, s='e')

    ax[1, 2].scatter(df_frame['x']/1000, df_frame['y']/1000, c=df_frame['ypred'], cmap='coolwarm',
                     vmin=df_train['dhdt_filt'].min(), vmax=df_train['dhdt_filt'].min()*-1)
    ax[1, 2].set_xlabel('x [km]')
    ax[1, 2].text(x=5, y=-997, s='f')

    cax = fig.add_axes([0.95, 0.2, 0.03, 0.5])
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label('Glacier elevation change [m/year]', labelpad=15)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('comparison_plot.png', bbox_inches='tight')
    plt.show()

    print('a = training data')
    print('b = k-NN')
    print('c = SE-ARD GP')
    print('d = original implementation')
    print('e = framework sparse GP')
    print('f = framework chunks')


if __name__ == '__main__':
    plot()

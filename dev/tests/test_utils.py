import matplotlib.pyplot as plt

def compare_signals(x1, y1, x2, y2):
    fig = plt.figure()

    num_subfigs = y1.shape[1]
    for j in range(num_subfigs):
        ax = fig.add_subplot(num_subfigs, 1, j+1)
        ax.plot(x1, y1[:, j], label="y1")
        ax.plot(x2, y2[:, j], label="y2")
        ax.legend()

    return fig

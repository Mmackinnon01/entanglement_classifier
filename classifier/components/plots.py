import matplotlib.pyplot as plt


def plotReservoir(time_log: dict):
    title = "Density Matrix Time Evolution of Reservoir"
    fig, ax = plotDensityMatrix(time_log, title)
    return fig


def plotReservoirAndSystem(time_log: dict):
    title = "Density Matrix Time Evolution of Reservoir and System"
    fig, ax = plotDensityMatrix(time_log, title)
    return fig


def plotDensityMatrix(time_log, title):
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=[15, 8])
    density_matrix_size = list(
        time_log.values())[0].shape[0]

    for i in range(density_matrix_size):
        ax.plot(
            list(time_log.keys()),
            [mat[i][i] for mat in list(time_log.values())],
            label=r"$\rho_{{{}{}}}$".format(i, i),
            alpha=0.7,
            linewidth=2,
        )
    fig.legend(loc="lower center", ncol=4)
    ax.set_title(title)

    return fig, ax


def plotExcitations(time_log):
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=[15, 8])
    no_expectations = len(list(
        time_log.values())[0])

    for i in range(no_expectations):
        ax.plot(
            list(time_log.keys()),
            [expectation[i] for expectation in list(time_log.values())],
            label=r"Node {}".format(i),
            alpha=0.7,
            linewidth=2,
        )
    fig.legend(loc="lower center", ncol=4)
    ax.set_title("Excitation Expectation Values for System")

    return fig, ax

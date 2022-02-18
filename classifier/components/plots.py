import matplotlib.pyplot as plt
import numpy as np


def plotReservoir(time_log: dict, off_diag=False):
    title = "Density Matrix Time Evolution of Reservoir"
    if off_diag:
        fig, ax = plotOffDiagonalElements(time_log, title)
    else:
        fig, ax = plotDiagonalElements(time_log, title)
    return fig


def plotReservoirAndSystem(time_log: dict, off_diag=False):
    title = "Density Matrix Time Evolution of Reservoir and System"
    if off_diag:
        fig, ax = plotOffDiagonalElements(time_log, title)
    else:
        fig, ax = plotDiagonalElements(time_log, title)
    return fig


def plotDiagonalElements(time_log, title):
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


def plotOffDiagonalElements(time_log, title):
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=[15, 8])
    density_matrix_size = list(
        time_log.values())[0].shape[0]

    for i in range(density_matrix_size):
        for j in range(density_matrix_size):
            if i != j:
                line = np.abs(np.array([mat[i][j]
                                        for mat in list(time_log.values())]))
                if line.any():
                    ax.plot(
                        list(time_log.keys()),
                        line,
                        label=r"$\rho_{{{}{}}}$".format(i, j),
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


def plotTotalExcitations(time_log):
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=[15, 8])

    ax.plot(
        list(time_log.keys()),
        [expectation[0] for expectation in list(time_log.values())],
        label="Sigma X Expectation",
        alpha=0.7,
        linewidth=2,
    )

    ax.plot(
        list(time_log.keys()),
        [expectation[0] for expectation in list(time_log.values())],
        label="Sigma Y Expectation",
        alpha=0.7,
        linewidth=2,
    )
    fig.legend(loc="lower center", ncol=4)
    ax.set_title("Total Excitation Expectation Values for System")

    return fig, ax


def plotSigmaCombinations(time_log):
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=[15, 8])

    for operator in list(list(time_log.values())[0].keys()):
        line = np.real(np.array([expectation[operator]
                                for expectation in list(time_log.values())]))
        if line.any():
            ax.plot(
                list(time_log.keys()),
                line,
                label=operator+"_real",
                alpha=0.7,
                linewidth=2,
            )

    for operator in list(list(time_log.values())[0].keys()):
        line = np.imag(np.array([expectation[operator]
                                for expectation in list(time_log.values())]))
        if line.any():
            ax.plot(
                list(time_log.keys()),
                line,
                label=operator+"_imag",
                alpha=0.7,
                linewidth=2,
            )
    fig.legend(loc="lower center", ncol=4)
    ax.set_title("Total Sigma Combination Expectation Values for System")

    return fig, ax

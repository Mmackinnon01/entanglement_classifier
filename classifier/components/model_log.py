import matplotlib.pyplot as plt
import numpy as np

from .partial_trace import partialTrace


class ModelLog:
    def __init__(self, run_resolution):
        self.time_log = {}
        self.partial_time_log = {}
        self.current_time = 0
        self.run_resolution = run_resolution

    def addLogEntry(self, entry):
        self.evalulateEntry(entry)
        self.time_log[self.current_time] = entry
        self.current_time += self.run_resolution

    def evalulateEntry(self, entry):
        validEigen = np.all(
            [val >= -1e-5 for val in np.linalg.eigvals(entry)])
        validTrace = round(np.trace(entry), 5) == 1

        if not validEigen:
            print(
                "Warning: entry at t={} has negative eigenvalues of {}".format(
                    round(self.current_time, 5),
                    [val for val in np.linalg.eigvals(entry) if val <= -1e-15]
                )
            )

        if not validTrace:
            print(
                "Warning: entry at t={} has trace of {}".format(
                    self.current_time, np.trace(entry)
                )
            )

    def traceSystem(self, system_nodes, basis):
        for key, density_matrix in self.time_log.items():
            reduced_matrix = self.trace(
                density_matrix, system_nodes, basis)
            self.evalulateEntry(reduced_matrix)
            self.partial_time_log[key] = reduced_matrix

    def trace(self, density_matrix, system_nodes, basis):
        for i in range(system_nodes):
            density_matrix = partialTrace(
                density_matrix, trace_system=0, basis=basis)
        return density_matrix

    def plot(self, plot_trace=True):
        if plot_trace:
            log = self.partial_time_log
        else:
            log = self.time_log
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=[15, 8])
        density_matrix_size = log[list(
            log.keys())[0]].shape[0]
        for i in range(density_matrix_size):
            ax.plot(
                list(log.keys()),
                [mat[i][i] for mat in list(log.values())],
                label=r"$\rho_{{{}{}}}$".format(i, i),
                alpha=0.7,
                linewidth=2,
            )
        fig.legend(loc="lower center", ncol=4)
        ax.set_title("Density Matrix Time Evolution")

        self.fig = fig

import matplotlib.pyplot as plt
import numpy as np


class ModelLog:
    def __init__(self, run_resolution):
        self.time_log = {}
        self.current_time = 0
        self.run_resolution = run_resolution

    def addLogEntry(self, entry):
        self.evalulateEntry(entry)
        self.time_log[self.current_time] = entry
        self.current_time += self.run_resolution

    def evalulateEntry(self, entry):
        validEigen = np.all([val >= 0 for val in np.linalg.eigvals(entry)])
        validTrace = round(np.trace(entry), 5) == 1

        if not validEigen:
            print(
                "Warning: entry at t={} has negative eigenvalues".format(
                    self.current_time
                )
            )

        if not validTrace:
            print(
                "Warning: entry at t={} has trace of {}".format(
                    self.current_time, np.trace(entry)
                )
            )

    def plot(self):
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=[10, 10])
        density_matrix_size = self.time_log[list(self.time_log.keys())[0]].shape[0]
        for i in range(density_matrix_size):
            ax.plot(
                list(self.time_log.keys()),
                [mat[i][i] for mat in list(self.time_log.values())],
                label=r"$\rho_{{{}{}}}$".format(i, i),
                alpha=0.7,
                linewidth=2,
            )
        fig.legend(loc="lower center", ncol=4)
        ax.set_title("Density Matrix Time Evolution")

        self.fig = fig

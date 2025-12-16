import numpy as np
import pyswarms as ps
from commonsetup import dataset, n_hidden, X_train, X_test, y_train, y_test, n_inputs, n_classes, activation, n_iteration
import time
import itertools
import pandas as pd
import seaborn as sns
import matplotlib


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_classes, activation):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.activation = activation

    def count_param(self):
        """Calculate the total number of parameters to optimize."""
        return (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_classes) + self.n_hidden + self.n_classes

    def generate_logits(self, x, data):
        """ 
        Parameters:
        x: one PSO slution (a list of variables, i.e. coordinates of a particle) 
        data: The train or test data to be predited

        At first, the function builds the network by cutting the values in x into the weights and 
        biases of the NN. Then it passes the data and performs activation to get the logits. 
        """
        ind1 = self.n_inputs * self.n_hidden
        W1 = x[0:ind1].reshape((self.n_inputs, self.n_hidden))
        ind2 = ind1 + self.n_hidden
        b1 = x[ind1:ind2].reshape((self.n_hidden,))
        ind3 = ind2 + self.n_hidden * self.n_classes
        W2 = x[ind2:ind3].reshape((self.n_hidden, self.n_classes))
        b2 = x[ind3:ind3 + self.n_classes].reshape((self.n_classes,))
        
        z1 = data.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = self.activation(z1)  # Activation in Layer 1
        logits = a1.dot(W2) + b2  # Pre-activation in Layer 2
        return logits

    def forward_prop(self, params, X_train, y_train):
        """Calculate the loss using forward propagation."""
        logits = self.generate_logits(params, X_train)
        # Stable softmax: shift by row max to avoid overflow, add small eps to avoid log(0)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        eps = 1e-12
        correct_logprobs = -np.log(probs[range(X_train.shape[0]), y_train] + eps)
        loss = np.sum(correct_logprobs) / X_train.shape[0]
        return loss    
    
    def predict(self, weights, data):
        """Predict the classes based on trained weights."""
        logits = self.generate_logits(weights, data)
        return np.argmax(logits, axis=1)

class PSOOptimizer:
    def __init__(self, nn, c1, c2, w, swarm_size, n_iterations, batchsize):
        self.nn = nn
        self.c1 = c1  # self confidence
        self.c2 = c2  # swarm confidence
        self.w = w # inertia (omega)
        self.swarm_size = swarm_size
        self.n_iterations = n_iterations
        self.batchsize = batchsize

    def fitness_function(self, X, X_train, y_train):
        """
        Parameters:
        X: 2-D array holding the PSO solutions to be evaluated by the fitness function
        X_train: Train set
        Y_train: target of the train set

        This is the fitness function used by the PSO, which is to be implemented (completed ) 
        by the strudents. The objective is understanding the concept of how PSO is applied in 
        this use case, namely optimizing a NN.

        Note that in each iteration of the PSO algorithm, a set of solutions are generated, 
        namely one solution by each particle. These are passed to this fittness function in the 
        parameter X. Since each solution is a list of numbers (the coordinates of a particle 
        position), X is a two-dimensional array.

        Note that each solution is used to setup the weights and biases of the network. 
        Therefor, what you shuld do here is performing the forward propagation each time
        using the solution and random batches of training data to return the resulting accuracies 
        in a one-dimensional list.

        To do this successfully, refer to and understand the functions forward_prop() and 
        generate_logits() that is called inside it.

        Note that the current implementation of the function is random, which will run
        but it will result in low accuracy ~ 1/n_classes.
        """

        particle_loss = []
        X_batch, y_batch = self.random_batch(X_train, y_train)

        for x in X:
            loss = self.nn.forward_prop(x, X_batch, y_batch)
            particle_loss.append(loss)

        return np.array(particle_loss)

    def random_batch(self, X_train, y_train):
        indices = np.random.choice(len(X_train), size=self.batchsize, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        return (X_batch, y_batch)
    
    def optimize(self, X_train, y_train):
        """Perform the PSO optimization."""
        dimensions = self.nn.count_param()
        optimizer = ps.single.GlobalBestPSO(n_particles=self.swarm_size, dimensions=dimensions,
                                            options={'c1': self.c1, 'c2': self.c2, 'w': self.w})
        cost, weights = optimizer.optimize(self.fitness_function, iters=self.n_iterations, verbose=False,
                                       X_train=X_train, y_train=y_train)
        return weights

###Strg + K, then Strg + C to comment; Strg + K, then Strg + U to uncomment.

# def main():
#     ####### PSO  Tuning ################
#     # Tune the PSO parameters here trying to outperform the classic NN 
#     # For more about these parameters, see the lecture resources
#     par_C1 = 0.1
#     par_C2 = 0.1
#     par_W = 0.1
#     par_SwarmSize = 100
#     batchsize = 200 # The number of data instances used by the fitness function

#     print ("############ you are using the following settings:")
#     print ("Number hidden layers: ", n_hidden)
#     print ("activation: ", activation[0])
#     print ("Number of variables to optimize: ", (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes)
#     print ("PSO parameters C1: ", par_C1, "C2: ", par_C2, "W: ", par_W, "Swarmsize: ", par_SwarmSize,  "Ieteration: ", n_iteration)
#     print ("\n")


#     # Initialize Neural Network and PSO optimizer
#     nn = NeuralNetwork(n_inputs, n_hidden, n_classes, activation[0])
#     pso = PSOOptimizer(nn, par_C1, par_C2, par_W, par_SwarmSize, n_iteration, batchsize)

#     # Perform optimization
#     weights = pso.optimize(X_train, y_train)

#     # Evaluate accuracy on the test set
#     y_pred = nn.predict(weights, X_test)
#     accuracy = (y_pred == y_test).mean()
#     print(f"Accuracy PSO-NN: {accuracy:.2f}")


#acc secs was manually recoded from classic-NN.py
def assign_acc_secs(dataset):
    if dataset == "cancer":
        acc = 0.62
        secs = 0.249
    elif dataset == "glass":
        acc = 0.7
        secs = 0.398
    elif dataset == "gamma":
        acc = 0.78
        secs = 20.171 
    return acc, secs

def run_pso(par_C1, par_C2, par_W, par_SwarmSize, n_iteration, batchsize):
    nn = NeuralNetwork(n_inputs, n_hidden, n_classes, activation[0])
    pso = PSOOptimizer(nn, par_C1, par_C2, par_W, par_SwarmSize, n_iteration, batchsize)

    t0 = time.time()
    # Perform optimization
    weights = pso.optimize(X_train, y_train)
    secs = time.time() - t0

    # Evaluate accuracy on the test set
    y_pred = nn.predict(weights, X_test)
    accuracy = (y_pred == y_test).mean()
    
    return accuracy, secs

def grouped_plot(df, best_ids=None):
    params = ["c1", "c2", "w", "sw", "iters", "bs"]
    ncols = 3
    nrows = (len(params) + ncols - 1) // ncols

    fig, axes = matplotlib.pyplot.subplots(nrows, ncols, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, plot_param in zip(axes, params):
        df["param_value"] = plot_param + "_" + df[plot_param].astype(str)
        df["param_value"] = pd.Categorical(df["param_value"])
        hue_order = list(df["param_value"].cat.categories)
        palette = sns.color_palette("tab20", n_colors=len(hue_order))

        sns.scatterplot(
            data=df, x="secs", y="acc",
            hue="param_value", hue_order=hue_order,
            palette=palette, ax=ax,
        )

        classic_acc, classic_secs = assign_acc_secs(dataset)

        classic_marker = ax.scatter(classic_secs, classic_acc, color="orange", edgecolor="black", zorder=5, label="classic-NN")
        ax.set_title(plot_param)
        ax.set_xlabel("secs")
        ax.set_ylabel("acc")
        handles, labels = ax.get_legend_handles_labels()
        if "classic-NN" not in labels:
            handles.append(classic_marker)
            labels.append("classic-NN")
        ax.legend(handles, labels, title=plot_param)

        if best_ids:
            for row in df.itertuples():
                if row.run_id in best_ids:
                    ax.text(row.secs, row.acc, str(row.run_id),
                            fontsize=5, ha="center", va="center")

    for ax in axes[len(params):]:
        ax.set_visible(False)
    fig.tight_layout()

    g = sns.relplot(
        data=df, x="secs", y="acc",
        hue="c1", style="c2", size="w",
        col="bs", col_wrap=2, kind="scatter",
        palette="tab10", alpha=0.8, height=3.5,
        legend=True
    )
    classic_acc, classic_secs = assign_acc_secs(dataset)

    # add classic point to every facet
    for ax in g.axes.flatten():
        cm = ax.scatter(classic_secs, classic_acc, color="orange",
                        edgecolor="black", zorder=5, label="classic-NN")
        ax.text(classic_secs, classic_acc, "classic-NN",
                            fontsize=5, ha="center", va="center")

    matplotlib.pyplot.show()



def repeat_and_plot(configs, n_repeats=10):
    """
    configs: list of (c1, c2, w, sw, iters, bs) tuples to test
    n_repeats: number of runs per config
    """
    records = []
    for cfg_id, (c1, c2, w, sw, iters, bs) in enumerate(configs):
        for r in range(n_repeats):
            acc, secs = run_pso(c1, c2, w, sw, iters, bs)
            records.append({
                "cfg_id": cfg_id,
                "repeat": r,
                "acc": acc,
                "secs": secs,
                "c1": c1, "c2": c2, "w": w, "sw": sw, "iters": iters, "bs": bs,
            })

    df_rep = pd.DataFrame(records)

    # summary stats
    summary = (
        df_rep
        .groupby(["cfg_id", "c1", "c2", "w", "sw", "iters", "bs"])
        [["acc", "secs"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "cfg_id", "c1", "c2", "w", "sw", "iters", "bs",
        "acc_mean", "acc_std", "secs_mean", "secs_std"
    ]

    # plot means with error bars
    matplotlib.pyplot.figure(figsize=(6,4))
    matplotlib.pyplot.errorbar(
        summary["secs_mean"], summary["acc_mean"],
        xerr=summary["secs_std"], yerr=summary["acc_std"],
        fmt="o", ecolor="gray", capsize=3, label=None
    )
    for _, row in summary.iterrows():
        matplotlib.pyplot.text(
            row.secs_mean, row.acc_mean, f"cfg{int(row.cfg_id)}",
            fontsize=7, ha="center", va="bottom"
        )
    matplotlib.pyplot.xlabel("secs")
    matplotlib.pyplot.ylabel("acc")
    matplotlib.pyplot.title(f"{n_repeats} repeats per config")
    matplotlib.pyplot.show()

    return df_rep, summary


def main():
    c1_grid = [0.5, 1.7]
    c2_grid = [0.5, 1.7]
    w_grid = [0.4, 0.9]
    swarmsize_grid = [30, 100]
    iter_grid = [30, 100]
    batch_grid = [int(len(X_train) * 0.5), int(len(X_train) * 0.8)]

    results = []
 
    for c1, c2, w, sw, iters, bs in itertools.product(c1_grid, 
                                                      c2_grid, 
                                                      w_grid, 
                                                      swarmsize_grid,
                                                      iter_grid,
                                                      batch_grid):
        acc, secs = run_pso(c1, c2, w, sw, iters, bs)
        results.append((acc, secs, c1, c2, w, sw, iters, bs))

    df = pd.DataFrame(
        results,
        columns=["acc", "secs", "c1", "c2", "w", "sw", "iters", "bs"]
    )
    df["combo"] = (
        "c1=" + df.c1.astype(str)
        + " | c2=" + df.c2.astype(str)
        + " | w=" + df.w.astype(str)
        + " | sw=" + df.sw.astype(str)
        + " | iters=" + df.iters.astype(str)
        + " | bs=" + df.bs.astype(str)
    )

    df = df.reset_index(drop=True)
    df["run_id"] = df.index

    corr_spearman = df[["acc", "secs", "c1", "c2", "w", "sw", "iters", "bs"]].corr(method="spearman")
    print(corr_spearman)

    best = df.nlargest(10, "acc")
    #print(best[["acc", "combo"]])
    #best.to_excel("../best_configuration.xlsx", index=False)

    best_ids = set(best["run_id"])
    grouped_plot(df, best_ids)

###     investigating spread ###
    configs = [
    (0.5, 0.5, 0.4, 30, 30, int(len(X_train) * 0.5)),
    (1.7, 1.7, 0.9, 100, 100, int(len(X_train) * 0.8)),
    ]
    df_rep, summary = repeat_and_plot(configs, n_repeats=10)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

def plotBounds(
    clf,
    X_train, y_train,
    X_test,  y_test,
    featureX, featureY,
    target_names,
    filename: str = "knn_boundaries.png",
    dpi: int = 300,
):
    """
    Plot 2-D KNN decision boundaries (uniform vs distance) and save to a file.

    Parameters
    ----------
    clf : sklearn Pipeline whose final step is named 'knn'
    X_train, y_train, X_test, y_test : pandas objects
    featureX, featureY : str – axis labels
    target_names : list-like – legend labels
    filename : str – output image path
    dpi : int – resolution of saved image
    """
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    for ax, weights in zip(axs, ("uniform", "distance")):
        clf.set_params(knn__weights=weights).fit(X_train, y_train)

        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X_test,
            response_method="predict",
            plot_method="pcolormesh",
            shading="auto",
            alpha=0.4,
            ax=ax,
        )

        scatter = disp.ax_.scatter(
            X_test.iloc[:, 0],
            X_test.iloc[:, 1],
            c=y_test,
            edgecolors="k"
        )
        ax.set(
            xlabel=featureX,
            ylabel=featureY,
            title=f"3-class KNN (k={clf[-1].n_neighbors}, weights='{weights}')"
        )
        handles, _ = scatter.legend_elements()
        ax.legend(handles, target_names, loc="lower left", title="Classes")

    plt.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)          # free memory
    print(f"Decision-boundary figure saved to: {filename}")

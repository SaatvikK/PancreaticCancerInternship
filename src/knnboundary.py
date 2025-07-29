import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

def plotBounds(
    clf,                   # already-fitted classifier or Pipeline
    X, y,                  # DataFrame or ndarray (n_samples, 2) | 1-D labels
    featureX, featureY,    # axis labels
    target_names=["Alive", "Dead"],          # legend labels
    filename: str = "knn_boundaries.png",
    dpi: int = 300,
):


  # Column accessor that handles DataFrame or ndarray
  col = lambda M, idx: M.iloc[:, idx] if hasattr(M, "iloc") else M[:, idx]

  # Locate the correct parameter name for "weights"
  params = clf.get_params()
  if "weights" in params:                 # bare estimator
    w_param = "weights"
  else:                                   # look for something like 'knn__weights'
    w_param = next(
      (k for k in params if k.endswith("__weights")), 
      None
    )
    if w_param is None:
      raise ValueError(
        "Could not find a 'weights' parameter on the supplied classifier."
      )

  # Preserve the original setting so we can restore it later
  original_weights = params[w_param]

  fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

  for ax, weights in zip(axs, ("uniform", "distance")):
    # Flip weighting strategy (no need to refit)
    clf.set_params(**{w_param: weights})

    disp = DecisionBoundaryDisplay.from_estimator(
      clf,
      X,
      response_method="predict",
      plot_method="pcolormesh",
      shading="auto",
      alpha=0.4,
      ax=ax,
    )

    scatter = disp.ax_.scatter(
      col(X, 0), col(X, 1),
      c=y,
      edgecolors="k"
    )

    ax.set(
      xlabel=featureX,
      ylabel=featureY,
      title=f"3-class KNN (k={clf.n_neighbors if hasattr(clf, 'n_neighbors') else clf[-1].n_neighbors}, "
        f"weights='{weights}')"
    )
    handles, _ = scatter.legend_elements()
    ax.legend(handles, target_names, loc="lower left", title="Classes")

  # Restore original weighting
  clf.set_params(**{w_param: original_weights})

  plt.tight_layout()
  fig.savefig(filename, dpi=dpi, bbox_inches="tight")
  plt.close(fig)
  print(f"Decision-boundary figure saved to: {filename}")

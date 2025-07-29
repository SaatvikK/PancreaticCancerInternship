import pandas as pd
import numpy as np
import sklearn as skl
#import tabulate

class Model:
  def __init__(self, preload = None): # algorithm = one of ["lrc", "knn", "svm", "nbc"], preload = Some SKL model if want.
    self.model = preload if preload is not None else None
    self.trained = False

  def train(self, X, Y):
    if not self.model:
      raise Exception("Please create the model first by calling `initModel(algorithm: str, preLoad)`")

    self.model = self.model.fit(X, Y)
    self.trained = True
  
  def test(self, X, Y):
    if not self.model: raise Exception("Please create the model first by calling `initModel(algorithm: str, preLoad)`")
    
    if not self.trained: raise Exception("Please train model first.")

    Ypred = self.model.predict(X)
    return self.compMetrics(Y, Ypred), Ypred
  
  def compMetrics(self, Y, Ypred, average="weighted"):
    """
    Return a dict with common classification metrics.
    """
    return {
        "accuracy": skl.metrics.accuracy_score(Y, Ypred),
        "precision": skl.metrics.precision_score(Y, Ypred, average=average, zero_division=0),
        "recall": skl.metrics.recall_score(Y, Ypred, average=average, zero_division=0),
        "f1": skl.metrics.f1_score(Y, Ypred, average=average, zero_division=0),
        "confusion_matrix": skl.metrics.confusion_matrix(Y, Ypred),
        "classification_report": skl.metrics.classification_report(Y, Ypred),
    }

  def ppMetrics(self, Ytrue, Ypred, metrics, classLabels = ["alive", "dead"]):
    """
    Pretty-print a metrics dictionary, now with row/column totals.

    Parameters
    ----------
    metrics : dict
        Output of your evaluate() helper.
    classLabels : list[str] | tuple[str], optional
        Names for the classes, in the same order as the encoded
        confusion-matrix axes.
    """
    # ── overall scalar scores ───────────────────────────────────
    print("\n📊  Overall scores")
    print("-" * 60)
    for key in ("accuracy", "precision", "recall", "f1"):
      print(f"{key.capitalize():<12}: {metrics[key]:>8.4f}")

    # ── full sklearn classification report ─────────────────────
    print("\n📝  Sklearn classification report")
    print("-" * 60)
    print(metrics["classification_report"])

    # ── confusion matrix with totals ───────────────────────────
    print("\n📝  Sklearn classification report")
    print("-" * 60)
    print(metrics["classification_report"])

    # ── pandas confusion matrix with totals ────────────────────
    # If you want nicer column names, remap Ytrue / Ypred first.
    if classLabels is not None:
      label_map = {lbl: name for lbl, name in zip(sorted(set(Ytrue)), classLabels)}
      Ytrue_print = pd.Series(Ytrue).map(label_map)
      Ypred_print = pd.Series(Ypred).map(label_map)
    else:
      Ytrue_print = Ytrue
      Ypred_print = Ypred

    cm_df = pd.crosstab(
      pd.Series(Ytrue_print, name="Actual"),
      pd.Series(Ypred_print, name="Predicted"),
      margins=True,                       # adds totals
      dropna=False                        # include any missing classes
    )

    print("\n🔀  Confusion matrix (pandas)")
    print("-" * 60)
    print(cm_df.to_string())

class knnModel(Model):
  def __init__(self, preload = None, neighbours = 5, metric = "manhattan"):
    super().__init__(preload=preload)
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:

      self.neighbours = neighbours
      self.metric = metric
      self.model = skl.neighbors.KNeighborsClassifier(
        n_neighbors=int(neighbours), metric=metric
      )

  # Convenience method (static so it can be reused elsewhere)
  @staticmethod
  def make_knn(neighbours=5, metric="manhattan"):
      return skl.neighbors.KNeighborsClassifier(
          n_neighbors=int(neighbours), metric=metric
      )
  
class lrcModel(Model):
  def __init__(self, preload = None):
    super().__init__(preload=preload)
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:

      self.model = skl.linear_model.LogisticRegression()

  # Convenience method (static so it can be reused elsewhere)
  @staticmethod
  def make_knn():
      return skl.linear_model.LogisticRegression()
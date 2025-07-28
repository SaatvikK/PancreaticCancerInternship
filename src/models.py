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
    return self.compMetrics(Y, Ypred)
  
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

  def ppMetrics(self, metrics):
    # ------------------------ #
    # 2️⃣  Pretty‑print summary #
    # ------------------------ #
    print("\n🔹 Overall metrics")
    print("-----------------")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k.capitalize():>9}: {metrics[k]:.3f}")

    # ------------------------ #
    # 3️⃣  Detailed class report#
    # ------------------------ #
    print("\n🔹 Classification report")
    print("------------------------")
    print(metrics["classification_report"].rstrip())

    # ------------------------ #
    # 4️⃣  Confusion matrix 🧩  #
    # ------------------------ #
    labels = [str(i) for i in range(len(metrics["confusion_matrix"]))]
    df_cm = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)

    print("\n🔹 Confusion matrix")
    print("-------------------")
    print(tabulate.tabulate(df_cm, headers="keys", tablefmt="github"))


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
  

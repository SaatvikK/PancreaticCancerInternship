from models.baseModel import Model
import xgboost as xgb
import sklearn as skl
import matplotlib.pyplot as plt

class knnModel(Model):
  def __init__(self, preload = None, neighbours = 5, metric = "manhattan"):
    super().__init__(preload=preload, algorithm = "knn")
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
    super().__init__(preload=preload, algorithm = "lrc")
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:

      self.model = skl.linear_model.SGDClassifier(loss = "log_loss", max_iter = 1, learning_rate = "optimal")

  # Convenience method (static so it can be reused elsewhere)
  @staticmethod
  def makeLRC():
      return skl.linear_model.LogisticRegression()

class svmModel(Model):
  def __init__(self, preload = None):
    super().__init__(preload = preload, algorithm = "svm")
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:

      # USING SGDCLASSIFIER WITH HINGE LOSS IS EQUIVALENT TO A LINEAR SVM.
      # We use SGDClassifier and hinge loss in order to have the `partial_fit` function - which allows us to record the learning curves.
      self.model = skl.linear_model.SGDClassifier(loss = "hinge", max_iter = 1, learning_rate = "optimal", average = True)

    # Convenience method (static so it can be reused elsewhere)
    @staticmethod
    def makeSVM():
        return skl.linear_model.LogisticRegression()

class dtcModel(Model):
  def __init__(self, preload = None):
    super().__init__(preload = preload, algorithm = "dtc")
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:
      self.model = skl.tree.DecisionTreeClassifier()

    # Convenience method (static so it can be reused elsewhere)
    @staticmethod
    def makeDTC():
        return skl.tree.DecisionTreeClassifier()

class rfcModel(Model):
  def __init__(self, preload = None):
    super().__init__(preload = preload, algorithm = "dtc")
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:
      self.model = skl.ensemble.RandomForestClassifier()

    # Convenience method (static so it can be reused elsewhere)
    @staticmethod
    def makeRFC():
        return skl.ensemble.RandomForestClassifier()

class xgbModel(Model):
  def __init__(self, preload = None):
    super().__init__(preload = preload, algorithm = "xgb")
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:
      self.model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)

    # Convenience method (static so it can be reused elsewhere)
    @staticmethod
    def makXGB():
        return xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    
class nbcModel(Model):
  def __init__(self, preload = None):
    super().__init__(preload = preload, algorithm = "nbc")
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:
      self.model = skl.naive_bayes.CategoricalNB()

    # Convenience method (static so it can be reused elsewhere)
    @staticmethod
    def makXGB():
        return skl.naive_bayes.CategoricalNB()
import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt

class Model:
  def __init__(self, preload = None, algorithm = "lrc"): # algorithm = one of ["lrc", "knn", "svm", "nbc"], preload = Some SKL model if want.
    self.model = preload if preload is not None else None
    self.trained = False
    self.algorithm = algorithm
  
  def calibrate(self, Xval, Yval, targetPrecision):
    # Now we instantiate the CalibratedClassifierCV object
    calibrator = skl.calibration.CalibratedClassifierCV(estimator = self.model, method = "isotonic", cv = "prefit")

    # fit the calibrator with the validation set
    calibrator.fit(Xval, Yval)
    
    # Now get the raw probabilities from the prediction with the X validation set.
    pVal = calibrator.predict_proba(Xval)[:,1]

    # Now we plot a precision-recall curve with respect to all possible thresholds, `targetPrecision`. 
    # This returns three lists: the precision, recall, and which threshold value is needed to achive them.
    prec, rec, thr = skl.metrics.precision_recall_curve(Yval, pVal)

    # Now we use a mask on `rec` by `prec` to find the threshold value that keeps the precision >= targetPrec but maximises recall.

    mask = (prec[:-1] >= targetPrecision)          # keep thresholds that meet the precision floor
    if len(rec[:-1][mask]) == 0: 
      return "Could not find a suitable threshold for the target precision. Lower the target precision."
  
    idx = np.argmax(rec[:-1][mask])                # among them, take the one with max recall
    print("RECALLS", rec)
    print("BEST REC", rec[idx])
    self.bestτ = thr[mask][idx]                         # that’s the actual numeric threshold τ*
    return calibrator

  def train(self, data, epochs = 10000, targetPrecision = 0.8):
    Xtr, Ytr, Xval, Yval = data["train"]["X"], data["train"]["Y"], data["validation"]["X"], data["validation"]["Y"]#
    classes = np.unique(Ytr)

    accuracies = []
    print(f"TRAINING {self.algorithm} MODEL WITH {Xtr.shape[0]} INSTANCES...")
    # Using a calibrator here to calibrate the probabilities.
    # Essentially trains the model such that at the end the version of the model that maximises recall and keeps precision as close to 0.8 as
    # possible is chosen.

    # First we fit with the normal model as usual:
    if self.algorithm in ["dtc", "rfc"]:
      self.model.fit(Xtr, Ytr)

    elif self.algorithm == "xgb":
      
      pos = int((Ytr == 1).sum())
      neg = int((Ytr == 0).sum())
      spw = neg / max(pos, 1)
      self.model.set_params(
        n_estimators=5000,        # big cap
        learning_rate=0.02,       # small step size
        eval_metric="aucpr",      # better for minority class than "error" (where accuracy = 1 - error)
        scale_pos_weight=spw,
        n_jobs=-1,
        enable_categorical = True
      )
      """
      self.model.set_params(
        n_estimators=1000,            # an upper bound
        eval_metric="error"           # classification error; accuracy = 1 - error
      )"""
      self.model.fit(Xtr, Ytr, eval_set=[(Xval, Yval)], verbose = True)

      # Get evals per round of xgb
      evals = self.model.evals_result_ if hasattr(self.model, "evals_result_") else self.model.evals_result()
      val_err = evals["validation_0"]["aucpr"]
      accuracies = [1.0 - e for e in val_err] # find accuracies
      self.graphEpochAcc(accuracies, len(accuracies)) # graph
    
    self.trained = True
    # Now calibrate:
    #calibrator = self.calibrate(Xval, Yval, targetPrecision)
    #if type(calibrator) == str: return calibrator # error message during calibration
    #self.model = calibrator
    return self.model

    # for linear model epoch training
    """
    else:
      for epoch in range(epochs):
        if epoch == 0:
          self.model.partial_fit(Xtr, Ytr, classes)
        else:
          self.model.partial_fit(Xtr, Ytr)

        Y = self.model.predict(Xval)
        accuracies.append(skl.metrics.accuracy_score(Yval, Y))
      self.graphEpochAcc(accuracies, epochs)
    """
  
  def test(self, data):
    X, Y = data["test"]["X"], data["test"]["Y"]
    if not self.model: raise Exception("Please create the model first by calling `initModel(algorithm: str, preLoad)`")
    if not self.trained: raise Exception("Please train model first.")

    if self.algorithm in ["dtc", "rfc", "nbc"] and hasattr(self, "bestτ"):
      pTest = self.model.predict_proba(X)[:,1]
      Ypred = (pTest >= self.bestτ).astype(int)
      return self.compMetrics(Y, Ypred), Ypred
    else:
      Ypred = self.model.predict(X)
      return self.compMetrics(Y, Ypred), Ypred
     
  def graphEpochAcc(self, accuracies, epochs):
    plt.plot(accuracies)
    plt.xlabel(f"Epoch({epochs})")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Change With Epoch Number")
    print("ALOGIRTHM:", self.algorithm)
    plt.savefig(f'../results/learning curves/{self.algorithm}_AccGraph.png')
    plt.close()

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
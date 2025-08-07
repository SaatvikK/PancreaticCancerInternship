import pandas as pd
import numpy as np
import sklearn as skl
import knnboundary as boundgraph
import matplotlib.pyplot as plt
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import tabulate

class Model:
  def __init__(self, preload = None, algorithm = "lrc"): # algorithm = one of ["lrc", "knn", "svm", "nbc"], preload = Some SKL model if want.
    self.model = preload if preload is not None else None
    self.trained = False
    self.algorithm = algorithm

  def train(self, data, epochs = 10000):
    Xtr, Ytr, Xval, Yval = data["train"]["X"], data["train"]["Y"], data["validation"]["X"], data["validation"]["Y"]#
    classes = np.unique(Ytr)

    if not self.model:
      raise Exception("Please create the model first by calling `initModel(algorithm: str, preLoad)`")

    accuracies = []
    if self.algorithm == "knn":
      self.model.fit(Xtr, Ytr)
      acc = skl.metrics.accuracy_score(Yval, self.model.predict(Xval))
      accuracies = [acc] * epochs

    else:
      for epoch in range(epochs):
        if epoch == 0:
          self.model.partial_fit(Xtr, Ytr, classes)
        else:
          self.model.partial_fit(Xtr, Ytr)

        Y = self.model.predict(Xval)
        accuracies.append(skl.metrics.accuracy_score(Yval, Y))
      self.graphEpochAcc(accuracies, epochs)

    self.trained = True
  
  def test(self, data):
    X, Y = data["test"]["X"], data["test"]["Y"]
    if not self.model: raise Exception("Please create the model first by calling `initModel(algorithm: str, preLoad)`")
    if not self.trained: raise Exception("Please train model first.")

    Ypred = self.model.predict(X)
    return self.compMetrics(Y, Ypred), Ypred
  
  def graphEpochAcc(self, accuracies, epochs):
    plt.plot(accuracies)
    plt.xlabel(f"Epoch({epochs})")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Change With Epoch Number")
    print("ALOGIRTHM:", self.algorithm)
    plt.savefig(f'../results/learning curves/{self.algorithm}_AccGraph.png')

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
    super().__init__(preload=preload, algorithm = "knn")
    # Only build a new estimator if nothing was pre‑loaded
    if self.model is None:

      self.neighbours = neighbours
      self.metric = metric
      self.model = skl.neighbors.KNeighborsClassifier(
        n_neighbors=int(neighbours), metric=metric
      )

  def plotClassBound(self, X, Y, featureX, featureY, target):
    boundgraph.plotBounds(self.model, X, Y, featureX, featureY)

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

      self.model = skl.linear_model.SGDClassifier(loss = "log_loss", max_iter = 1, learning_rate = "optimal", average = True)

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


############ DNN ############
"""
class dnnModel(nn.Module, Model):
  def __init__(self, inputSize, hiddenSizes, outputSize, name = "DNN", preload = None):
    self.inputSize = inputSize
    self.hiddenSizes = hiddenSizes
    self.outputSize = outputSize
    self.name = name

    nn.Module.__init__(self)
    Model.__init__(preload = preload, algorithm = "dnn") # Model not passed into baseModel class to avoid reaching the maximum recursion depth.

    # Create a list to hold all layers
    layers = []
    # Define the layer sizes: start with inputSize, followed by your hidden layer sizes
    layer_sizes = [self.inputSize] + self.hiddenSizes

    # Loop to create each hidden layer with a ReLU activation
    for i in range(len(layer_sizes) - 1):
      layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
      layers.append(nn.Sigmoid())

    # Add the output layer (we don't usually follow the output layer with an activation like ReLU
    # if you're performing classification with CrossEntropyLoss, as it expects raw logits)
    layers.append(nn.Linear(hiddenSizes[-1], self.outputSize))
    
    # Use nn.Sequential to combine all the layers into a single network
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    # The forward method defines how the data passes through the network layers
    return self.network(x)

  def predict(self, Xtest):
    Xtest = torch.tensor(Xtest.values, dtype=torch.float32)
    self.eval() # setting model to evaluation mode
    with torch.no_grad():
      outputs = self(Xtest)
      _, preds = torch.max(outputs, 1)
    
    preds = preds.numpy()
    print(type(preds))
    #predsdf = pd.DataFrame(preds, columns=["diagnosis"])
    return preds

  def validationTest(self, Xval, Yval):
    preds = self.predict(Xval)
    accuracy = skl.metrics.accuracy_score(Yval, preds)
  
  def dnnTrain(self, data, noEpochs = 5, α = 0.001):
    Xtr, Ytr, Xval, Yval = data["train"]["X"], data["train"]["Y"], data["validation"]["X"], data["validation"]["Y"]
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.RMSprop(self.parameters(), lr=α)

    self.accuracies = []
    for epoch in range(noEpochs):  # Training for 5 epochs
      # Simulated dummy data (replace these with actual data during implementation)
      inputs = torch.tensor(Xtr.values, dtype=torch.float32)
      labels = torch.tensor(Ytr.values, dtype=torch.long)

      # Zero the gradients from the previous iteration
      self.optimizer.zero_grad()
      
      # Forward pass: Compute predicted outputs by passing inputs to the model
      outputs = self(inputs)
      
      # Calculate the loss
      loss = self.criterion(outputs, labels)
      
      # Backward pass: Compute gradient of the loss with respect to model parameters
      loss.backward()
      
      # Update parameters
      self.optimizer.step()
      
      # Print the loss for each epoch to track progress
      print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

      # now do validation for this epoch
      self.accuracies.append(self.validationTest(Xval, Yval))
    
    self.graphEpochAcc(self.accuracies, epochs = noEpochs)
    return [True, None, self]
"""
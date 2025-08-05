# hi
import sqlite3 as sql
import pandas as pd
import preprocess as preproc
import models
import aggregation
import numpy as np
import sklearn as skl
import pprint as pp

class Main:
  def __init__(self, dbPath = "../data-v2-working/Data", dbname = "dataset"):
    self.dbPath = dbPath
    self.dbname = dbname
    self.aggor = aggregation.Aggregator(self.dbPath, self.dbname)
  
  def splitSet(self, mergedTable: pd.DataFrame, valPerc = 0.1, testPerc = 0.2, trainPerc = 0.7):
    # 1. Split set into X (features) and Y (target)
    # 1a. Split into X and Y
    X = mergedTable.drop(columns=[self.preprocor.label])
    Y = mergedTable[self.preprocor.label]
    npX = self.preprocor.convToNumpy(X)
    npY = self.preprocor.convToNumpy(Y)
    
    # 2. First split test from training
    Xtrain, Xtest, Ytrain, Ytest = skl.model_selection.train_test_split(
      npX, npY, test_size=testPerc, random_state=42, stratify=npY
    )

    # 3. Then split validation from training
    Xtrain, Xval, Ytrain, Yval = skl.model_selection.train_test_split(Xtrain, Ytrain, test_size = valPerc, random_state = 42, stratify = Ytrain)

    return {
      "train": {"X": Xtrain, "Y": Ytrain},
      "test": {"X": Xtest, "Y": Ytest},
      "validation": {"X": Xval, "Y": Yval}
    }


  def main(self, projLabel = "VITALSTATUS"):
    # Instantiate a preprocessor
    self.preprocor = preproc.Preprocessor(label = projLabel)

    # Using the aggregator's in-built querier, get the patient table: `select * from patient`
    patient = self.aggor.dbRequester(self.aggor.workingPath, overwrite = False)
    
    # Now aggregate on all selected tables
    regimen, tumourgene = self.aggregate()
    
    # Now join all tables based on PATIENTID:
    # We choose the `inner` method as we ONLY want patients who have an instance in ALL tables.
    mergedTable = patient \
    .merge(regimen, on='PATIENTID', how='inner') \
    .merge(tumourgene, on='PATIENTID', how='inner')
    print("=========== FINAL MERGED TABLE ===========")
    print(mergedTable)

    data = self.splitSet(mergedTable)

    # Train the models
    print("===================== K NN MODEL =====================")
    knn = models.knnModel(preload = None, neighbours = 5, metric = "manhattan")
    knn.train(data["train"]["X"], data["train"]["Y"])
    res, Ypred = knn.test(data["test"]["X"], data["test"]["Y"])
    knn.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead", "D4", "X5"])

    print("===================== LRC MODEL =====================")
    lrc = models.knnModel(preload = None)
    lrc.train(data["train"]["X"], data["train"]["Y"])
    res, Ypred = lrc.test(data["test"]["X"], data["test"]["Y"])
    lrc.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead", "D4", "X5"])

  def aggregate(self):
    regimen = self.aggor.regimenPatient() # regiment -> patient
    #gene = self.aggor.geneTumour() # gene -> tumour
    #res = self.aggor.joinTables(self.aggor.workingPath, table1 = "tumour", table2 = "gene", common = "TUMOURID")
    # now after gene -> tumour aggregation we can aggregate (gene, tumour) -> patient
    tumour = self.aggor.tumourPatient() # (gene,tumour) -> patient
    return [regimen, tumour]


mainC = Main()
mainC.main("VITALSTATUS")
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
    .merge(regimen, on='PATIENTID', how='outer') \
    .merge(tumourgene, on='PATIENTID', how='outer')
    print("=========== FINAL MERGED TABLE ===========")
    print(mergedTable)

    # Split into X and Y
    X = mergedTable.drop(columns=[self.preprocor.label])
    Y = mergedTable[self.preprocor.label]
    npX = self.preprocor.convToNumpy(X)
    npY = self.preprocor.convToNumpy(Y)

    # Split into training and testing sets (80/20)
    X_train, X_test, Y_train, Y_test = skl.model_selection.train_test_split(
        npX, npY, test_size=0.2, random_state=42, stratify=npY
    )

    # Train the models
    print("===================== K NN MODEL =====================")
    knn = models.knnModel(preload = None, neighbours = 5, metric = "manhattan")
    knn.train(X_train, Y_train)
    res, Ypred = knn.test(X_test, Y_test)
    knn.ppMetrics(Y_test, Ypred, res, classLabels = ["alive", "dead", "D4", "X5"])

    print("===================== LRC MODEL =====================")
    lrc = models.knnModel(preload = None)
    lrc.train(X_train, Y_train)
    res, Ypred = lrc.test(X_test, Y_test)
    lrc.ppMetrics(Y_test, Ypred, res, classLabels = ["alive", "dead", "D4", "X5"])

  def aggregate(self):
    regimen = self.aggor.regimenPatient() # regiment -> patient
    #gene = self.aggor.geneTumour() # gene -> tumour
    #res = self.aggor.joinTables(self.aggor.workingPath, table1 = "tumour", table2 = "gene", common = "TUMOURID")
    # now after gene -> tumour aggregation we can aggregate (gene, tumour) -> patient
    tumour = self.aggor.tumourPatient() # (gene,tumour) -> patient
    return [regimen, tumour]


mainC = Main()
mainC.main("VITALSTATUS")
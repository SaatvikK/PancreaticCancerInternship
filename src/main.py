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
  def __init__(self, dbPath = "../data-v2-working/Data/dataset.db"):
    self.dbPath = dbPath
  
  def main(self, projLabel = "VITALSTATUS"):
    self.setPreprocor(label = projLabel)

    patient = self.readSQLDB(tableName = "patient")
    regimen, tumour = self.aggregate()
    
    # Now join all tables:
    # We choose the `inner` method as we ONLY want patients who have an instance in ALL tables.
    mergedTable = patient \
    .merge(regimen, on='PATIENTID', how='inner') \
    .merge(tumour, on='PATIENTID', how='inner')
    # Split into X and Y
    X = mergedTable.drop(columns=[self.preprocor.label])
    Y = mergedTable[self.preprocor.label]
    npX = self.preprocor.convToNumpy(X)
    npY = self.preprocor.convToNumpy(Y)

    # Split into training and testing sets (80/20)
    X_train, X_test, Y_train, Y_test = skl.model_selection.train_test_split(
        npX, npY, test_size=0.2, random_state=42, stratify=npY
    )
    model = models.knnModel(preload = None, neighbours = 5, metric = "manhattan")

    model.train(X_train, Y_train)
    res = model.test(X_test, Y_test)
    #pp.pprint(res)
    model.ppMetrics(res)

  def readSQLDB(self, tableName = "patient") -> pd.DataFrame:
    with sql.connect(self.dbPath) as connection:
      req = "select * from " + tableName.lower()
      res = pd.read_sql_query(req, connection)
      return res
  
  def setPreprocor(self, label = "VITALSTATUS"):
    self.preprocor = preproc.Preprocessor(label = label)
    return self.preprocor

  def aggregate(self):
    aggor = aggregation.Aggregator(self.dbPath)
    regimen = aggor.regimenPatient()
    tumour = aggor.tumourPatient()
    return [regimen, tumour]


mainC = Main()
mainC.main("VITALSTATUS")
# hi
import sqlite3 as sql
import pandas as pd
import preprocess as prepoc
import models

with sql.connect("../data-v2-working/Data/dataset.db") as con:
  sql = """
  SELECT * FROM regimen WHERE 
  """
  result = pd.read_sql_query(sql, con)
  print(result)


class Main:
  def __init__(self, dbPath = "../data-v2-working/Data/dataset.db"):
    self.dbPath = dbPath
  
  def main(self, projLabel = "VITALSTATUS"):
    self.setPreprocor(label = projLabel)

    table = self.readSQLDB(tableName = "patient")
    # Split into X and Y
    X = table[:, self.preprocor.label]
    Y = table[self.preprocor.label]
    npX = self.preprocor.convToNumpy(X)
    npY = self.preprocor.convToNumpy(Y)

    model = models.knnModel(preload = None, neighbours = 5, metric = "manhattan")

    model.train(npX, npY)
    



  def readSQLDB(self, tableName = "patient") -> pd.DataFrame:
    with sql.connect(self.dbPath) as connection:
      req = "select * from " + tableName.lower()
      res = pd.read_sql_query(sql, connection)
      return res
  
  def setPreprocor(self):
    self.preprocor = preproc.Preprocessor(label = "VITALSTATUS")
    return self.preprocor

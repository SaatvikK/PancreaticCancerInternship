import pandas as pd
import sklearn as skl
import numpy as np
import glob
import random as rand

class Preprocessor:
  def __init__(self, dirpath = "../data-v2-working/Data", label = ""):
    if dirpath[len(dirpath)-1] in ["/", "\\"]: raise Exception("`dirpath` cannot end with / or \\.")
    self.dirpath = dirpath
    self.dfs = self.openCSVs()

    # N.B. Tables with NO clear primary key are NOT included.
    self.ignoreCols = {
      "label": "VITALSTATUS",
      "primaryKeys": {
        "patient": "PATIENTID",
        "tumour": "TUMOURID",
        "geneid": "GENEID",
        "regimen": "MERGED_REGIMEN_ID",
        "outcome": "MERGED_REGIMEN_ID",
        "cycle": "MERGED_CYCLE_ID",
        "drug-detail": "MERGED_DRUG_DETAIL_ID",
        "episode": "RADIOTHERAPYEPISODEID",
        "prescription": "PRESCRIPTIONID,"
      }

    }

    pass
  
  def openCSVs(self):
    dfs = []
    paths = glob.glob(self.dirpath + "/*.csv")
    for path in paths:
      df = pd.read_csv(path)
      dfs.append(df)
    
    return dfs

  def featureSelection(self):
    fsdfs = []

    for df in self.dfs:
      newDF = self.someFeatureSelection(df)
      fsdfs.append(newDF)
      self.dfs.remove(df)
      



  def someFeatureSelection(df): # placeholder
    return df
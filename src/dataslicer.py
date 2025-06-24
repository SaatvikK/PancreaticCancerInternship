# ============= Info ============= #
# Created 23/06/2025
# Last updated 23/06/2025

# For isolating pancreatic tumours (C25x) in the sim_av_tumour.csv table by ICD-10/O2 3char.
# ================================ #

import pandas as pd
import os
import re 

class DataSlicer:
  def __init__(
      self, tablesPath = "", 
      tumourCode = "C25" # The ICD-10/O2 3-character code for each tumour. C25 for pancreatic malignant tumours.
    ):
    self.tablesPath = tablesPath
    self.code = tumourCode
    self.fulldfs = {} # ALL the data
    self.dfNames = []
    self.slicedDFs = {} # Data of the specific tumour/cancer.

  """
  Reads the desired table and converts to a DF. It adds the DF to `self.fulldfs` under a user-specified pseudonym.
  """
  def getTable(self, tableName = "", dfName = ""):
    self.fulldfs[dfName] = pd.read_csv(f'{self.tablesPath}/{tableName}')
    return self.fulldfs[dfName]

  def sliceTumourDF(self, dfName = "", colName = ""):
    df = self.fulldfs[dfName]
    sliced = df[df[colName] == self.code]
    self.sliced[dfName] = sliced
    return sliced
  
  def saveToCSV(self):
    path = "../data-v2-working/Data"
    keys = list(self.slicedDFs.keys())
    for key in keys:
      df = self.slicedDFs[key]
      df.to_csv(index = False, path_or_buf = f"{path}/{key}.csv")


  
  def createSlice(self):
    # First, for all CSVs in /Data, get the tables
    tables = os.listdir(self.tablesPath) # get everything inside the tables dir - in /Data there is only files, so no need to worry about dirs.
    tables = ["sim_av_tumour.csv", "sim_av_patient.csv", "sim_av_gene.csv"]

    for table in tables:
      # Quickly determine the DF name by isolating the important information in the table's file name.
      underPos = [match.start() for match in re.finditer("_", table)] # Search for the last underscore (underPos[1])
      theDFName = table[underPos[1]+1:len(table)-4] # Take a slice from the laster underscore to the .csv
      self.dfNames.append(theDFName)
      print(theDFName)
      df = self.getTable(tableName = table, dfName = theDFName)
      print(df)

    # Now, we slice each table following the order of relations provided in the Simulacrum's User Guide.

    # ================================ Tumour ================================ #
    # We first have to slice tumour.
    print("================================ Tumour ================================")
    df = self.fulldfs["tumour"]
    sliced = df[df.SITE_ICD10_O2_3CHAR == self.code]
    self.slicedDFs["tumour"] = sliced
    # ======================================================================== #

    # =============================== Patient =============================== #
    print("=============================== Patient ===============================")
    tumourdf = self.slicedDFs["tumour"]
    patientdf = self.fulldfs["patient"]
    patients = tumourdf.PATIENTID.tolist()
    slicedPatients = pd.DataFrame(columns=patientdf.columns) # create a new, empty DF to populate, with the same column names as patients.

    for i in range(len(patients)):
      patient = patients[i]
      instance = patientdf[patientdf.PATIENTID == patient] # get the instance in the patient table using the patient id.
      slicedPatients.loc[len(slicedPatients)] = instance.values.flatten().tolist()  # Add the patient to the new patient table for only patients with panc tumours.
      list(filter((patient).__ne__, patients)) # remove the patientt from the patient list.
      print(f"Done patient {patient} ({i}/{len(patients)-1})")

    self.slicedDFs["patient"] = slicedPatients # add sliced patients to dict.
    # ======================================================================== #

    # ================================= Gene ================================= #
    print("================================= Gene =================================")
    
    tumourdf = self.slicedDFs["tumour"]
    genedf = self.fulldfs["gene"]
    tumours = tumourdf.TUMOURID.tolist()
    slicedGenes = genedf[genedf.TUMOURID.isin(tumours)]
    self.slicedDFs["gene"] = slicedGenes # add sliced patients to dict.
    # ======================================================================== #

    self.saveToCSV()




slicer = DataSlicer("../data-v2-original/Data", tumourCode = "C25")
slicer.createSlice()


#pancreatic.to_csv(index=False, path_or_buf = "panc.csv")
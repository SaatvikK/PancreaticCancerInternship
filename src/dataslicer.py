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

  def oneToManySlicing(self, oneTable = "patient", manyTable = "gene", oneCol = "PATIENTID", manyCol = "GENEID", printOut = True):
    if printOut: print("=================================", manyTable.capitalize(), "=================================")
    onedf = self.slicedDFs[oneTable]
    manydf = self.fulldfs[manyTable]
    oneColList = onedf[oneCol].tolist()
    slicedMany = manydf[manydf[manyCol].isin(oneColList)]
    self.slicedDFs[manyTable] = slicedMany # add sliced patients to dict.
    if printOut: print(slicedMany)
    return slicedMany
  
  def createSlice(self):
    # First, for all CSVs in /Data, get the tables
    tables = os.listdir(self.tablesPath) # get everything inside the tables dir - in /Data there is only files, so no need to worry about dirs.
    tables = [
      "sim_av_tumour.csv", "sim_av_patient.csv", "sim_av_gene.csv", "sim_sact_regimen.csv", "sim_rtds_episode.csv", 
      "sim_sact_outcome.csv", "sim_sact_cycle.csv", "sim_sact_drug-detail.csv", "sim_rtds_prescription.csv"
    ]

    for table in tables:
      # Quickly determine the DF name by isolating the important information in the table's file name.
      underPos = [match.start() for match in re.finditer("_", table)] # Search for the last underscore (underPos[1])
      theDFName = table[underPos[1]+1:len(table)-4] # Take a slice from the laster underscore to the .csv
      self.dfNames.append(theDFName)
      print(theDFName)
      df = self.getTable(tableName = table, dfName = theDFName)
      print(df)
    print(list(self.fulldfs.keys()))
    # Now, we slice each table following the order of relations provided in the Simulacrum's User Guide.

    # ================================ Tumour ================================ #
    # We first have to slice tumour.
    print("================================ Tumour ================================")
    df = self.fulldfs["tumour"]
    sliced = df[df.SITE_ICD10_O2_3CHAR == self.code]
    self.slicedDFs["tumour"] = sliced
    # ======================================================================== #

    # =============================== Patient =============================== #
    # Tumour -> Patient, many to one relationship
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
    # Tumour -> Gene, one to many relationship
    self.oneToManySlicing(oneTable = "tumour", manyTable = "gene", oneCol = "TUMOURID", manyCol = "TUMOURID")
    # ======================================================================== #

    # ================================ Regimen =============================== #
    # Patient -> Regimen, one to many relationship
    self.oneToManySlicing(oneTable = "patient", manyTable = "regimen", oneCol = "PATIENTID", manyCol = "ENCORE_PATIENT_ID")

    # ======================================================================== #

    # ================================ Episode =============================== #
    # Patient -> Episode, one to many relationship
    self.oneToManySlicing(oneTable = "patient", manyTable = "episode", oneCol = "PATIENTID", manyCol = "PATIENTID")
    # ======================================================================== #

    # ================================ Outcome =============================== #
    # Regimen -> Outcome, one to many relationship
    self.oneToManySlicing(oneTable = "regimen", manyTable = "outcome", oneCol = "MERGED_REGIMEN_ID", manyCol = "MERGED_REGIMEN_ID")
    # ======================================================================== #

    # ================================= Cycle ================================ #
    # Regimen -> Outcome, one to many relationship
    self.oneToManySlicing(oneTable = "regimen", manyTable = "cycle", oneCol = "MERGED_REGIMEN_ID", manyCol = "MERGED_REGIMEN_ID")
    # ======================================================================== #

    # ============================== Drug Detail ============================= #
    # Regimen -> Outcome, one to many relationship
    self.oneToManySlicing(oneTable = "cycle", manyTable = "drug-detail", oneCol = "MERGED_CYCLE_ID", manyCol = "MERGED_CYCLE_ID")
    # ======================================================================== #

    # ============================== Prescription ============================ #
    # Episode -> Prescription, one to many relationship
    print("============================== Prescription ============================")
    episodedf = self.slicedDFs["episode"]
    prescriptiondf = self.fulldfs["prescription"]

    # Define the columns that form the composite key for the join
    key_columns = [
        'PATIENTID',
        'RADIOTHERAPYEPISODEID',
        'ATTENDID',
        'LINKCODE'
    ]

    # Use a 'right' join to keep all original prescription columns,
    # but only for rows that have a matching key in episodedf.
    # The 'inner' join is often preferred and clearer for this filtering task.
    slicedPresciptions = pd.merge(
        left=episodedf[key_columns], # Only the keys are needed from the 'one' side
        right=prescriptiondf,
        on=key_columns,
        how='inner' # 'inner' join ensures only matching records are kept
    )

    self.slicedDFs["prescription"] = slicedPresciptions # add sliced patients to dict.
    print(slicedPresciptions)
    # ======================================================================== #




############### DRIVER CODE ###############
slicer = DataSlicer("../data-v2-original/Data", tumourCode = "C61")
slicer.createSlice()
slicer.saveToCSV()
############### IMPORTS ###############
import sqlite3 as sql
import pandas as pd
#######################################

class Aggregator:
  def __init__(self, dbPath):
    self.dbPath = dbPath
    self.checkDB(dbPath)
    self.aggregations = {}
  
  # Method to check if the db actually exists.
  def checkDB(self, dbPath):
    try:
      with sql.connect(dbPath) as con:
        query = """
        SELECT * FROM patient 
        """
        result = pd.read_sql_query(query, con)
        print(result)
    except Exception as e:
      raise Exception("Error in opening DB. Please check that the DB has been created with the preprocess.Preprocessor.convToSQL() method: \n" + str(e))
  
  # Sends a query request to the aforementioned DB.
  def dbRequester(self, path = "../data-v2-working/Data/dataset.db", query = "select * from patient") -> pd.DataFrame:
    with sql.connect(path) as connection:
      return pd.read_sql_query(query, connection)

  # Aggregates the regimen table to the patient level.
  def regimenPatient(self):
    # This method aggregates rhe regimen table on to the patient level.
    # This SQL query will take every regimen row for each unique patient id and attempt to 
    # aggregate the columns using the method that makes most sense.
    query = """
      select
        ENCORE_PATIENT_ID as PATIENTID,
        HEIGHT_AT_START_OF_REGIMEN,
        avg(WEIGHT_AT_START_OF_REGIMEN) as AVE_WEIGHT,

        max(INTENT_OF_TREATMENT) as INTENT_OF_TREATMENT,
        min(START_DATE_OF_REGIMEN) as FIRST_START_DATE,
        max(START_DATE_OF_REGIMEN) as LAST_START_DATE,
        COUNT(MERGED_REGIMEN_ID) as NUM_REGIMENS
      
      from
        regimen
      
      group by
        ENCORE_PATIENT_ID
    """
    result = self.dbRequester(self.dbPath, query)
    self.aggregations["regimen->patient"] = result
    print("======== REGIMEN -> PATIENT ========")
    print(result)
    return result
  
  # Aggregates the tumour table to the patient level.
  def tumourPatient(self):
    # Tumour -> Patient aggregation
    # Count number of tumours per patient.
    # Get max T/N/M_BEST per patient
    # Use min age as this is the age when the first tumour was found.
    # Use max gleason prim/sec as this is the most severe.

    query = """
      select
        PATIENTID as PATIENTID,
        count(TUMOURID) as NUM_TUMOURS,
        max(T_BEST) as MAX_T_BEST,
        max(N_BEST) as MAX_N_BEST,
        max(M_BEST) as MAX_M_BEST,
        max(STAGE_BEST) as MAX_STAGE_BEST,
        max(GRADE) as GRADE,
        min(AGE) as AGE,
        max(GLEASON_PRIMARY) as GLEASON_PRIMARY,
        max(GLEASON_SECONDARY) as GLEASON_SECONDARY
      
      from
        tumour

      group by
        PATIENTID

    """
    result = self.dbRequester(self.dbPath, query)
    self.aggregations["tumour->patient"] = result
    return result

   

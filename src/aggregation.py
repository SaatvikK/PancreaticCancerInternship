############### IMPORTS ###############
import sqlite3 as sql
import pandas as pd
import shutil
#######################################

class Aggregator:
  def __init__(self, path, dbname):
    dbPath = path + "/" + dbname + ".db"
    self.path = path
    self.dbname = dbname
    self.dbPath = dbPath
    self.checkDB(dbPath)
    self.aggregations = {}
  

    working = self.path + "/" + dbname + "_working.db"
    self.workingPath = working

    # Copy and rename in one step
    shutil.copy(self.dbPath, working)

  # ============================ SQL COMMANDS ============================ #

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
  def dbRequester(self, path = "../data-v2-working/Data/dataset.db", query = "select * from patient", overwrite=False, tableName=None):
    with sql.connect(path) as connection:
      if overwrite:
        if not tableName:
          raise ValueError("overwrite=True requires tableName")

        scratch = f"tmp_{tableName}"

        swap_script = f"""
        begin;
        /* 1. build the new table under a temporary name */
        drop table if exists {scratch};
        create table {scratch} as
        {query};

        /* 2. drop the old table only after the new one is ready */
        drop table if exists {tableName};

        /* 3. rename the new table to the original name */
        alter table {scratch} rename to {tableName};
        commit;
        """
        connection.executescript(swap_script)

        return pd.read_sql_query(f"SELECT * FROM {tableName}", connection)

      # default behaviour
      return pd.read_sql_query(query, connection)

  def joinTables(self, path="../data-v2-working/Data/dataset.db", table1 = "patient", table2 = "tumour", common = "PATIENTID") -> pd.DataFrame:
    query = f"""
      select *
      from {table1}
      join {table2} using ({common})
    """
    return self.dbRequester(path, query = query, overwrite = True, tableName = table1)

  # ========================= AGGREGATION METHODS ========================= #
  # Aggregates the regimen table to the patient level.
  def regimenPatient(self):
    # This method aggregates rhe regimen table on to the patient level.
    # This SQL query will take every regimen row for each unique patient id and attempt to 
    # aggregate the columns using the method that makes most sense.
    query = """
      select
        ENCORE_PATIENT_ID as PATIENTID,
        avg(HEIGHT_AT_START_OF_REGIMEN) as AVE_HEIGHT,
        avg(WEIGHT_AT_START_OF_REGIMEN) as AVE_WEIGHT,
        max(INTENT_OF_TREATMENT) as INTENT_OF_TREATMENT,
        min(DATE_DECISION_TO_TREAT) as FIRST_DECISION_TO_TREAT,
        max(DATE_DECISION_TO_TREAT) as LAST_DECISION_TO_TREAT,
        min(START_DATE_OF_REGIMEN) as FIRST_START_DATE,
        max(START_DATE_OF_REGIMEN) as LAST_START_DATE,
        COUNT(MERGED_REGIMEN_ID) as NUM_REGIMENS
      
      from
        regimen
      
      group by
        ENCORE_PATIENT_ID
    """
    result = self.dbRequester(self.workingPath, query, overwrite = False, tableName = "regimen")
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
    result = self.dbRequester(self.workingPath, query, overwrite = False, tableName = "tumour")
    self.aggregations["tumour->patient"] = result
    print("======== (GENE?, TUMOUR) -> PATIENT ========")
    print(result)
    return result

  # Aggregates the tumour table to the patient level.
  def geneTumour(self):
    # Tumour -> Patient aggregation
    # Count number of tumours per patient.
    # Get max T/N/M_BEST per patient
    # Use min age as this is the age when the first tumour was found.
    # Use max gleason prim/sec as this is the most severe.

    query = """
      select
        TUMOURID as TUMOURID,
        PATIENTID as PATIENTID,
        count(GENEID) as NUM_GENES,
        count(distinct GENE) as GENE_TYPES,
        sum(COUNT_TESTS) as NUM_GENE_TESTS,
        sum(COUNT_RESULTS) as NUM_TEST_RESULTS,
        min(DATE_OVERALL_TS) as FIRST_DATE_TS,
        max(DATE_OVERALL_TS) as LAST_DATE_TS
      from
        gene

      group by
        TUMOURID

    """
    result = self.dbRequester(self.workingPath, query, overwrite = True, tableName = "gene")
    self.aggregations["gene->tumour"] = result
    print("======== GENE -> TUMOUR ========")
    print(result)
    return result
   

import pandas as pd
import sklearn as skl
import numpy as np
import glob, os
import random as rand
import sqlite3 as sql
import pathlib as pali
import json

class Preprocessor:
  def __init__(self, dirpath = "../data-v2-working/Data", label = ""):
    if dirpath[len(dirpath)-1] in ["/", "\\"]: raise Exception("`dirpath` cannot end with / or \\.")
    self.dirpath = dirpath
    self.dfs, self.dfnames = self.openCSVs()
    self.label = label
    # N.B. Tables with NO clear primary key are NOT included.
    self.ignoreCols = {
      "label": self.label,
      "primaryKeys": {
        "patient": "PATIENTID",
        "tumour": "TUMOURID",
        "geneid": "GENEID",
        "regimen": "MERGED_REGIMEN_ID",
        "outcome": "MERGED_REGIMEN_ID",
        "cycle": "MERGED_CYCLE_ID",
        "drug-detail": "MERGED_DRUG_DETAIL_ID",
        "episode": "RADIOTHERAPYEPISODEID",
        "prescription": "PRESCRIPTIONID"
      }

    }

    # Define the continuous features (we want to fill these in with interpolation/imputation, NOT mode)
    self.contFeatures = {
      "drug-detail": ["ACTUAL_DOSE_PER_ADMINISTRATION"], "prescription": ["RTACTUALDOSE"], "regimen": ["HEIGHT_AT_START_OF_REGIMEN", "WEIGHT_AT_START_OF_REGIMEN", "LINK_NUMBER"]
    }

    # Date cols
    self.dateFeatures = {} # table : feature name

  # ================================================== NaNs Handling ================================================== #
  def findNAs(self, deleteMajMissing = True, fillRestMissing = True, showRatios = True): # deleteMajMissing = Delete Majority (> 50%) Missing - i.e. if the majority of observations are missing, delete the feature.
    for i in range(len(self.dfs)):
      df = self.dfs[i]
      dfname = self.dfnames[i]
      try: 
        if dfname == "episode" or dfname == "prescription": df = df.drop(["ATTENDID"], axis = 1);
      except: pass;
      sumNaNs = df.isna().sum()
      sumNaNs, ratios = self.ratioMissing(sumNaNs, df.shape[0], show = showRatios)
      print("======================================== DF:", dfname, "========================================")
      print(sumNaNs)
      # If we want to delete any feature with > 50% of its observations missing
      #print("HIUHFIHFHIFHIFIHFHIFHIFHIFHIFIH")
      if deleteMajMissing and dfname != "patient":
        #print("NOT PATIENT")
        print(" ")
        print("Deleting all features with a missing rate of > 50%...")
        df = self.deleteNaNFeatures(ratios, df)
      
      # The vast majority of the features are boolean, or discrete. Very few continuous. So we can just fill the rest with the mode observation.
      if fillRestMissing:
        print(" ")
        print("Filling remaining NaNs with mode value...")
        df = self.fillNaNsMode(df, dfname)
      self.dfs[i] = df
      print("===================================================================================================================")

  def fillNaNsMode(self, df, dfname):
    dfNamesForContFeatures = self.contFeatures.keys()
    
    # Check to see if DF as a continuous feature
    if dfname in dfNamesForContFeatures:
      features = self.contFeatures[dfname]
      # Iterate through continuous features for the targetted table and interpolate any missing values of that feature.
      for feature in features:
        df[feature] = df[feature].interpolate(method = "linear")

    features = df.columns.values.tolist()
    for feature in features:
      # Since we've already filled the NaNs for the continuous features, they're not flagged in the .fillna() search, 
      # so they won't be interfered with.
      df[feature] = df[feature].fillna(df[feature].mode()[0])
    return df

  def deleteNaNFeatures(self, ratios, df):
    for feature, ratio in ratios.items():
      if ratio > 0.5:
        df = df.drop([feature], axis = 1)
    
    return df

  def ratioMissing(self, nanSum: pd.Series, numInst: int, show = True):
    if numInst == 0:
      raise Exception("Total number of instances cannot be zero.")
    
    ratio = nanSum/numInst # num missing for this feature / total number of instances
    return [nanSum.astype(str) + " (" + ratio.round(2).astype(str) + ")", ratio] if show == True else [nanSum, ratio]
  # ================================================================================================================== #
  
  # ================================================== Feature Selection ================================================== #
  def manualFS(self, desiredFeatures: dict) -> list:
    """
    desiredFeatures format:
    {
      "dfname": [<feature names as strings>]
    }
    """

    selectedDFs = list(desiredFeatures.keys()) # get the dfnames 
    # Loop through each DF that has at least one desired feature.

    print("========== BEGINNING MANUAL FEATURE SELECTION ==========")
    for i in range(len(selectedDFs)):
      dfname = selectedDFs[i]
      features = desiredFeatures[dfname]
      print("dfname:", dfname, ", features: ", features)
      dfIdx = self.dfnames.index(dfname)  # get the relevant df
      df = self.dfs[dfIdx]

      allFeatures = df.columns.values.tolist()

      if set(features) - set(allFeatures): raise Exception("Error: Not all the desired features selected for the df " + dfname + " are in the df. Please verify.")

      # Add ONLY the selected features to a new df.
      newdf = df.loc[:, features].copy(deep=True)

      # Deep copy to create a new variable
      self.dfs[dfIdx] = newdf.copy(deep = True)
      print(newdf)
      print("Done", dfname)
    
    # Names the caller *wants* to keep
    wanted = set(desiredFeatures.keys())

    # Build a list of indices for frames we will delete
    toDelete = [i for i, name in enumerate(self.dfnames) if name not in wanted]

    # Remove from the end to avoid index shifts
    for i in reversed(toDelete):
        del self.dfnames[i]
        del self.dfs[i]

    return self.dfs

  def featureSelection(self, method = "variance", manualFeatures = {}):
    self.numpyTables = [None]*len(self.dfs)

    if method == "manual":
      if manualFeatures == {}: raise Exception("You've chosen the manual feature selection option but have left the parameter empty. Please format your dicitonary as: manualFeaturs = {'dfname': [<list of feature names>], ...} ")

      # Do FS
      dfs = self.manualFS(desiredFeatures = manualFeatures)

      # Convert to numpy
      for i in range(len(dfs)):
        self.numpyTables[i] = self.convToNumpy(dfs[i])

    if method == "variance":
      for i in range(len(self.dfs)):
        df = self.dfs[i]
        dfname = self.dfnames[i]

        # Variance FS is unsupervised so no splitting is required.
        res = self.FeatSelectVar(df)
        print("=============================", dfname, "=============================")
        print("For df", dfname, "variances are:", res[1], "for features", res[2])
        print(" ")
        print("Number of features BEFORE:", df.shape[1], ", number of features AFTER:", res[0].shape[1])
        print("======================================================================")
        self.numpyTables[i] = res[0]

    return self.numpyTables

  def FeatSelectVar(self, X, th = 0.2):
    selector = skl.feature_selection.VarianceThreshold(threshold=th)
    return [selector.fit_transform(X), selector.variances_, selector.feature_names_in_]
  # ======================================================================================================================= #

  # ================================================== Discrete Mapping ================================================== #
  def mappingCheck(self, feature, dfname, datapoint):
    """True → map this column; False → leave it untouched."""
    i = self.dfnames.index(dfname)
    col = self.dfs[i][feature]      # quick handle

    # Reject dates (dtype or name cue)
    if pd.api.types.is_datetime64_any_dtype(col) or "DATE" in feature.upper():
        return False

    # Reject numeric dtypes
    if (
        pd.api.types.is_numeric_dtype(col)                              # real numeric dtype
        or pd.to_numeric(col.dropna(), errors="coerce")    # try cast non-missing rows
            .notna()
            .all()                                        # all rows converted OK
    ):
        return False

    # Reject declared continuous features
    if feature in self.contFeatures.get(dfname, []):
        return False

    # Reject label & primary-key columns
    #if feature == self.ignoreCols.get("label"):
    #    return False
    
    if feature == self.ignoreCols.get("primaryKeys", {}).get(dfname):
        return False

    # Discrete/Categorical -> map it
    return True
  
  def mapMaker(self, feature: pd.Series, fname, tablename) -> dict:
    uniqueVals = feature.unique() # Get each unique value that an observation of this feature can have.
    mapDict = {} # The map - actual_value : assigned_num

    for i in range(len(uniqueVals)):
       mapDict[uniqueVals[i]] = i
    

    # Write map to file for referencing later.
    data = {}
    try:
      with open("../data-v2-working/maps/" + tablename + ".json") as JsonFile:
        data = json.load(JsonFile)
    except FileNotFoundError as e:
      data = {}
    except json.JSONDecodeError as e:
      data = {}
    
    data[fname] = mapDict
    with open("../data-v2-working/maps/" + tablename + ".json", 'w') as outfile:
      json.dump(data, outfile)

    return mapDict

  def mapper(self):
    for i in range(len(self.dfs)):
      df, dfname = self.dfs[i], self.dfnames[i]
      features = df.columns.values.tolist()
      for feature in features:
        datapoint = df[feature].sample(1, random_state=None).iat[0]
        res = self.mappingCheck(feature, dfname, datapoint)
        print("res for feature", feature, "in df", dfname, ":", res)
        if res:
          featureMap = self.mapMaker(df[feature], feature, dfname)
          #print("map for feature", feature, "in df", dfname, ":", featureMap)
          df[feature] = df[feature].map(featureMap)
      
      self.dfs[i] = df
    # delete d4, d5, x5
    print("Removing D5, D4, and X5 class label names...")
    print("Remaining class label names:")
    print(self.deleteClassNames(namesToDel = [2, 3, 4, 5, 6, 7]))
  
  def deleteClassNames(self, namesToDel = []): #namesToDel is a list of numbers
    idx = self.dfnames.index("patient")
    df = self.dfs[idx]
    if len(namesToDel) == 0: raise Exception("`namesToDel` list is empty.")
    mask = ~df[self.label].isin(namesToDel)
    pruned = df.loc[mask].copy()   # one deep copy is plenty
    self.dfs[idx] = pruned
    return pruned[self.label].unique()

  # ====================================================================================================================== #

  # ==================================================== Date Features =================================================== #
  def findDateCols(self):
    for i in range(len(self.dfs)):
      df = self.dfs[i]
      dfname = self.dfnames[i]
      features = df.columns.values.tolist()
      for feature in features:
        if "DATE" in feature or "date" in feature:
          if dfname in self.dateFeatures.keys():
            self.dateFeatures[dfname].append(feature)
          else:
            self.dateFeatures[dfname] = [feature]

    print("DATE FEATURES:", self.dateFeatures)
    return self.dateFeatures
  
  def convToUnix(self):
    dfsWithDates = self.dateFeatures.keys()
    print("NUMBER OF DFS WITH DATE FEATURES:", len(dfsWithDates))
    for datedDF in dfsWithDates:
      df = self.dfs[ self.dfnames.index(datedDF) ]

      for feature in self.dateFeatures[datedDF]:
        print("Feature:", feature, "in df", datedDF)
        dtS = pd.to_datetime(df[feature], format="%Y-%m-%d", errors="coerce")

        # Ensure the timestamps are in UTC
        dtS = dtS.dt.tz_localize("UTC")

        # Convert to Unix (pandas stores ns, so divide by 1e+9)
        unix = (dtS.astype("int64") // 1_000_000_000).astype("Int64")
        df[feature] = unix
        print(df[feature])
  # ====================================================================================================================== #
  
  def saveToCSV(self):
    try:
      for fname in glob.glob(self.dirpath + "/*"):
        os.remove(fname)
    except: pass;

    for i in range(len(self.dfs)):
      path = self.dirpath + "/" + self.dfnames[i] + ".csv"
      df = self.dfs[i]
      df.to_csv(index = False, path_or_buf = path)
  
  def openCSVs(self):
    dfs = []
    dfnames = []
    paths = glob.glob(self.dirpath + "/*.csv")
    for path in paths:
      df = pd.read_csv(path)
      dfs.append(df)
      dfnames.append(path[len(self.dirpath+"/"):len(path)-4])
    
    return [dfs, dfnames]
  
  def printFeatType(self):
    for i in range(len(self.dfs)):
      df = self.dfs[i]
      dfname = self.dfnames[i]
      featuresNames = df.columns.values.tolist()
      for featureName in featuresNames:
        feature = df[featureName]
        print("Feature", featureName, "from df", dfname, "is of type", feature.dtype)    

  # Method to "guess" the data type of a given feature (assuming it has been number mapped if it's categorical)
  def detFeatureType(self, col: pd.Series, colName):
    nunqVals = col.dropna().unique()
    
    if "ID" in colName: return "id"
    if "DATE" in colName: return "date"

    if set(nunqVals).issubset({0, 1}): return "bool"

    ratioNunqs = len(nunqVals) / len(col)

    if ratioNunqs < 0.2 or len(nunqVals) < 20:
      return "discr"
    
    else:
      return "cont"

  def buildFeatureDict(self):
    dictionary = {}
    for dfname, df in zip(self.dfnames, self.dfs):
      dictionary[dfname] = {}
      for feature in df.columns:
        dictionary[dfname][feature] = self.detFeatureType(df[feature], feature)
    return dictionary

  @staticmethod
  def convToNumpy(df):
    return df.to_numpy()
  
  # Convert the CSVs into SQL
  def convToSQL(self):
    folder   = pali.Path(self.dirpath)           # where your CSVs live
    dbPath  = pali.Path(self.dirpath + "/dataset.db")

    with sql.connect(dbPath) as con:
        for csvFile in folder.glob("*.csv"):
            table = csvFile.stem              # filename without .csv
            print(f"Loading {csvFile.name} → {table}")

            df = pd.read_csv(csvFile)
            df.to_sql(table, con, if_exists="replace", index=False)

            # auto-index any column that ends with "id"
            for col in df.columns:
                if col.lower().endswith("id"):
                    ix = f"idx_{table}_{col}"
                    con.execute(f"CREATE INDEX IF NOT EXISTS {ix} ON {table}({col});")

preprocor = Preprocessor(label = "VITALSTATUS")
#preprocor.findNAs(deleteMajMissing = True, fillRestMissing = True, showRatios = True)
#print("drfhigodrfhgndfsohgisdoihgnsdftiohsfhtghjisdfgoijhnsdpfohpsjodfgnpjoxdfgnjopsfgnop[jsfgjopnsfgjop[nsfdgpjo[nsfdo[jgn[sjofdnj[oksdfgfnjo[sdfgno[]]]]]]]]")
#preprocor.findNAs(deleteMajMissing = False, fillRestMissing = False, showRatios = False)
#preprocor.mapper()
#preprocor.findDateCols()
#preprocor.convToUnix()
"""
preprocor.featureSelection(method = "manual", manualFeatures = {
  "patient": [
    "PATIENTID",
    "ETHNICITY",
    "DEATHCAUSECODE_1A",
    "DEATHCAUSECODE_1B",
    "DEATHCAUSECODE_1C",
    "DEATHCAUSECODE_2",
    "VITALSTATUS"
  ],
  "tumour": [
    "TUMOURID",
    "PATIENTID",
    "T_BEST",
    "N_BEST",
    "M_BEST",
    "STAGE_BEST",
    "GRADE",
    "AGE",
    "GLEASON_PRIMARY",
    "GLEASON_SECONDARY"
  ],
  "regimen": [
    "ENCORE_PATIENT_ID",
    "MERGED_REGIMEN_ID",
    "HEIGHT_AT_START_OF_REGIMEN",
    "WEIGHT_AT_START_OF_REGIMEN",
    "INTENT_OF_TREATMENT",
    "START_DATE_OF_REGIMEN"
  ]
})
"""
#preprocor.printFeatType()
#preprocor.saveToCSV()
preprocor.convToSQL()

#import pprint
#pprint.pprint(preprocor.buildFeatureDict())
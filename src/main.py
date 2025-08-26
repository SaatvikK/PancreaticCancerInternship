# hi
import sqlite3 as sql
import pandas as pd
import preprocessing.preprocess as preproc
import models.baseModel as baseModel
import models.models as models
import preprocessing.aggregation as aggregation
import numpy as np
import sklearn as skl
import pprint as pp
import sqlite3 as sql

class Main:
  def __init__(self, dbPath = "../data-v2-working/Data", dbname = "dataset"):
    self.dbPath = dbPath
    self.dbname = dbname
    self.aggor = aggregation.Aggregator(self.dbPath, self.dbname)

  def main(self, projLabel = "VITALSTATUS"):
    # Instantiate a preprocessor
    self.preprocor = preproc.Preprocessor(label = projLabel)

    # Aggregrate all tables
    mergedTable = self.aggregate()

    ############################## DROPPING FEATURES BASED ON RFE FEATURE RANKING ##############################
    # Drop the last 5 common features, as outlined by feature ranking (main README.md under #badRecallInvestigation)
    mergedTable.drop(columns=["NUM_TUMOURS",  "LINKNUMBER", "GLEASON_PRIMARY", "AVE_WEIGHT", "GRADE"], inplace = True)

    # Drop the next 5
    mergedTable.drop(columns=["GLEASON_SECONDARY", "LAST_START_DATE", "AVE_HEIGHT", "ETHNICITY", "AGE"], inplace = True)
    print(mergedTable)

    # write the table to the db for later usage
    conn = sql.connect("../data-v2-working/Data/dataset_working.db")
    mergedTable.to_sql("mergedTable", conn, if_exists="append", index=False)  # creates if missing, appends otherwise
    conn.close()

    ############################## NOW IMPUTE THE MISSING OBSERVATIONS (SOME NULL DUE TO LEFT-AGGREGATION) ##############################
    mergedTable = self.manFillInRegimen(mergedTable)
    imputer = self.chooseImputer(imputeMethod = "simple")
    data = self.preprocor.splitSet(mergedTable)
    self.impute(imputer, data)

    ############################## DATA DEBIASING ##############################
    data["train"]["X"], data["train"]["Y"] = self.preprocor.debias(data["train"]["X"], data["train"]["Y"])

    ############################## RUN MODELS ##############################
    self.runModels(data)

  def runModels(self, data):
    # Train the models
    print("===================== K NN MODEL =====================")
    knn = models.knnModel(preload = None, neighbours = 5, metric = "manhattan")
    knn.train(data)
    res, Ypred = knn.test(data)
    knn.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    """
    print("===================== LRC MODEL =====================")
    lrc = models.lrcModel(preload = None)
    lrc.train(data)
    res, Ypred = lrc.test(data)
    lrc.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    print("Feature Ranking")
    print("-" * 60)
    print("LRC Feature Ranking with RFE:")
    selector = skl.feature_selection.RFE(lrc.model, step = 1, n_features_to_select = 1)
    selector.fit(data["train"]["X"], data["train"]["Y"])
    print(selector.ranking_)

    print("===================== SVM MODEL =====================")
    svm = models.svmModel(preload = None)
    svm.train(data)
    res, Ypred = svm.test(data)
    svm.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    print("Feature Ranking")
    print("-" * 60)
    print("SVM Feature Ranking with RFE:")
    selector = skl.feature_selection.RFE(svm.model, step = 1, n_features_to_select = 1)
    selector.fit(data["train"]["X"], data["train"]["Y"])
    print(selector.ranking_)
    """
    print("===================== DTC MODEL =====================")
    dtc = models.dtcModel(preload = None)
    dtc.train(data)
    res, Ypred = dtc.test(data)
    dtc.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    print("Feature Ranking")
    print("-" * 60)
    print("DTC Feature Ranking with RFE:")
    selector = skl.feature_selection.RFE(dtc.model, step = 1, n_features_to_select = 1)
    #selector.fit(data["train"]["X"], data["train"]["Y"])
    #print(selector.ranking_)

    print("===================== RFC MODEL =====================")
    rfc = models.rfcModel(preload = None)
    rfc.train(data)
    res, Ypred = rfc.test(data)
    rfc.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    print("Feature Ranking")
    print("-" * 60)
    print("RFC Feature Ranking with RFE:")
    selector = skl.feature_selection.RFE(rfc.model, step = 1, n_features_to_select = 1)
    #selector.fit(data["train"]["X"], data["train"]["Y"])
    #print(selector.ranking_)
    
    print("===================== XGB MODEL =====================")
    xgb = models.xgbModel(preload = None)
    xgb.train(data)
    res, Ypred = xgb.test(data)
    xgb.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    print("Feature Ranking")
    print("-" * 60)
    #print("RFC Feature Ranking with RFE:")
    #selector = skl.feature_selection.RFE(xgb.model, step = 1, n_features_to_select = 1)
    #selector.fit(data["train"]["X"], data["train"]["Y"], )
    #print(selector.ranking_)

    print("===================== NBC MODEL =====================")
    nbc = models.nbcModel(preload = None)
    nbc.train(data)
    res, Ypred = nbc.test(data)
    nbc.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    print("Feature Ranking")
    print("-" * 60)
    #print("RFC Feature Ranking with RFE:")
    #selector = skl.feature_selection.RFE(nbc.model, step = 1, n_features_to_select = 1)
    #selector.fit(data["train"]["X"], data["train"]["Y"], )
    #print(selector.ranking_)

  def aggregate(self):
    # Using the aggregator's in-built querier, get the patient table: `select * from patient`
    patient = self.aggor.dbRequester(self.aggor.workingPath, overwrite = False)
    
    # Now aggregate on all selected tables

    regimen = self.aggor.regimenPatient() # regiment -> patient
    #gene = self.aggor.geneTumour() # gene -> tumour
    #res = self.aggor.joinTables(self.aggor.workingPath, table1 = "tumour", table2 = "gene", common = "TUMOURID")
    # now after gene -> tumour aggregation we can aggregate (gene, tumour) -> patient
    tumourgene = self.aggor.tumourPatient() # (gene,tumour) -> patient
    

    # Now join all tables based on PATIENTID:
    # We choose the `inner` method as we ONLY want patients who have an instance in ALL tables.
    mergedTable = patient \
    .merge(regimen, on='PATIENTID', how='left') \
    .merge(tumourgene, on='PATIENTID', how='left')
    print("=========== FINAL MERGED TABLE ===========")
    return mergedTable
    
  def chooseImputer(self, imputeMethod = "simple"):
    # Initialize the KNNImputer with k=2 (you can adjust k as needed)
    imputer = None
    match(imputeMethod):
      case "simple":
        imputer = skl.impute.SimpleImputer(strategy="mean")
      case "knn":
        imputer = skl.impute.KNNImputer(n_neighbors=2)
      case "iter":
        imputer = skl.impute.IterativeImputer(initial_strategy = "mean")
      case _:
        raise Exception("Please choose valid imputation option.")
    
    return imputer
  
  def impute(self, imputer, data):
    imputer.fit(data["train"]["X"]) # ONLY fit the imputer to X train

    # Transform each set
    for split in ["train", "validation", "test"]:
      data[split]["X"] = imputer.transform(data[split]["X"])
    
    return data

  def manFillInRegimen(self, df):
    # the algorithm of this function is as follows:
    
    # 1. get ages, heights, weights as lists
    # 2. call the findAvePerAge() function
    #   -> this function uses `ages`, `heights`, and `weights` to find the average height & weight per age-range 
    #   -> dictionary provided in the func
    #   -> it returns a list of dictionaries of average values per age-range
    # 3. loop through each row of the final merged table.
    # 4. check if the age of a specific row has NA for AVE_HEIGHT and AVE_WEIGHT
    # 5. using this `averages` list, fill in the AVE_HEIGHT and AVE_WEIGHT for this row
    # 6. repeat until looped through full dataset



    def findAvePerAge(ages, heights, weights):
      """
      Bin (age, height, weight) into ranges and compute per-bin averages.

      Age bins (inclusive): [16–24], [25–34], [35–44], [45–54], [55–64], [65–74], [75+]

      Returns: list of dicts:
        { "range": (low, high_or_None), "avg_height": float|None,
          "avg_weight": float|None, "count": int }
      """
      # first remove all pd.na-type elements from `heights` and `weights`
      nonaheights, nonaweights = [], []
      for i in range(len(heights)):
        if not np.isnan(heights[i]): nonaheights.append(heights[i])
        if not np.isnan(weights[i]): nonaweights.append(weights[i])

      # overwrite old with new
      heights, weights = nonaheights, nonaweights

      # define bins
      bins = [
        {"range": (16, 24), "heights": [], "weights": []},
        {"range": (25, 34), "heights": [], "weights": []},
        {"range": (35, 44), "heights": [], "weights": []},
        {"range": (45, 54), "heights": [], "weights": []},
        {"range": (55, 64), "heights": [], "weights": []},
        {"range": (65, 74), "heights": [], "weights": []},
        {"range": (75, None), "heights": [], "weights": []},
      ]

      for age, height, weight in zip(ages, heights, weights):
        for Bin in bins:
          low, high = Bin["range"]
          if (age >= low) and (high is None or age <= high):
            # only add if both height & weight present
            Bin["heights"].append(float(height))
            Bin["weights"].append(float(weight))
            break

      aves = []
      for Bin in bins:
        n = len(Bin["heights"])  # if we appended, lengths match
        avgh = sum(Bin["heights"]) / n if n else None
        avgw = sum(Bin["weights"]) / n if n else None
        aves.append({
          "range": Bin["range"],
          "avg_height": avgh,
          "avg_weight": avgw,
          "count": n,
        })
      
      return aves

    # We try to get the three columns - but it's possible they've been removed due to feature ranking
    # So we check if they still exist - if not we just move on and dont run this method.
    try:
      ages = df.AGE.values.tolist()
      heights = df.AVE_HEIGHT.values.tolist()
      weights = df.AVE_WEIGHT.values.tolist()

      averages = findAvePerAge(ages, heights, weights)

      df = df.reset_index(drop=True)

      for idx, row in df.iterrows():
        #print(row)
        for avg in averages:
          min_age, max_age = avg["range"]

          # Handle no upper limit case
          if (max_age is None and row.AGE >= min_age) or (min_age <= row.AGE <= max_age):
            if pd.isna(row.AVE_HEIGHT):
              df.at[idx, "AVE_HEIGHT"] = avg["avg_height"]
              #print(df.at[idx, "AVE_HEIGHT"])
              #print(avg["avg_height"])
            if pd.isna(row.AVE_WEIGHT):
              # Here you can choose BMI=22 or BMI=25
              df.at[idx, "AVE_WEIGHT"] = avg["avg_weight"]
              #print(df.at[idx, "AVE_WEIGHT"])
              #print(avg["avg_weight"])
            break  # Found the right range, no need to check others
    except:
      pass
    
    def decToTreat(df, which = "FIRST"):
      colNameTreat = f'{which}_DECISION_TO_TREAT'
      colNameDateReg = f'{which}_START_DATE'
      medianDecisionToTreat = int(df[colNameTreat].dropna().median())
      df[colNameTreat].fillna(medianDecisionToTreat, inplace = True)

      existingDelay = (df[colNameDateReg] - df[colNameTreat]).dropna()
      medianDelay = int(existingDelay.median()) # delay between the decision to treat and the start of regimen

      df[colNameDateReg].loc[df[colNameDateReg].isna()] = df[colNameTreat] + medianDelay
      return df
    if "FIRST_START_DATE" in df.columns.values and "FIRST_DECISION_TO_TREAT" in df.columns.values:
      df = decToTreat(df, which = "FIRST")
    if "LAST_START_DATE" in df.columns.values and "LAST_DECISION_TO_TREAT" in df.columns.values:
      df = decToTreat(df, which = "LAST")
    return df

mainC = Main()
mainC.main("VITALSTATUS")
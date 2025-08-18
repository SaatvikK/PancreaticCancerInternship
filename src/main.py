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
  
  def splitSet(self, mergedTable: pd.DataFrame, valPerc = 0.1, testPerc = 0.2, trainPerc = 0.7):
    # 1. Split set into X (features) and Y (target)
    # 1a. Split into X and Y
    X = mergedTable.drop(columns=[self.preprocor.label])
    Y = mergedTable[self.preprocor.label]
    npX = self.preprocor.convToNumpy(X)
    npY = self.preprocor.convToNumpy(Y)
    
    # 2. First split test from training
    Xtrain, Xtest, Ytrain, Ytest = skl.model_selection.train_test_split(
      npX, npY, test_size=testPerc, random_state=42, stratify=npY
    )

    # 3. Then split validation from training
    Xtrain, Xval, Ytrain, Yval = skl.model_selection.train_test_split(Xtrain, Ytrain, test_size = valPerc, random_state = 42, stratify = Ytrain)

    return {
      "train": {"X": Xtrain, "Y": Ytrain},
      "test": {"X": Xtest, "Y": Ytest},
      "validation": {"X": Xval, "Y": Yval}
    }


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
    .merge(regimen, on='PATIENTID', how='left') \
    .merge(tumourgene, on='PATIENTID', how='left')
    print("=========== FINAL MERGED TABLE ===========")
    mergedTable = self.manFillInRegimen(mergedTable)
    print(mergedTable)
    imputer = self.chooseImputer(imputeMethod = "simple")
    data = self.splitSet(mergedTable)
    self.impute(imputer, data)


    # Train the models
    print("===================== K NN MODEL =====================")
    knn = models.knnModel(preload = None, neighbours = 5, metric = "manhattan")
    knn.train(data)
    res, Ypred = knn.test(data)
    knn.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])

    print("===================== LRC MODEL =====================")
    lrc = models.lrcModel(preload = None)
    lrc.train(data)
    res, Ypred = lrc.test(data)
    lrc.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])
    print("Feature Ranking")
    print("-" * 60)
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
    selector = skl.feature_selection.RFE(svm.model, step = 1, n_features_to_select = 1)
    selector.fit(data["train"]["X"], data["train"]["Y"])
    print(selector.ranking_)

    #dnn.ppMetrics(data["test"]["Y"], Ypred, res, classLabels = ["alive", "dead"])

  def aggregate(self):
    regimen = self.aggor.regimenPatient() # regiment -> patient
    #gene = self.aggor.geneTumour() # gene -> tumour
    #res = self.aggor.joinTables(self.aggor.workingPath, table1 = "tumour", table2 = "gene", common = "TUMOURID")
    # now after gene -> tumour aggregation we can aggregate (gene, tumour) -> patient
    tumour = self.aggor.tumourPatient() # (gene,tumour) -> patient
    return [regimen, tumour]

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
          if pd.isna(row.AVE_WEIGHT):
            # Here you can choose BMI=22 or BMI=25
            df.at[idx, "AVE_WEIGHT"] = avg["avg_weight"]
          break  # Found the right range, no need to check others

    
    def decToTreat(df, which = "FIRST"):
      colNameTreat = f'{which}_DECISION_TO_TREAT'
      colNameDateReg = f'{which}_START_DATE'
      medianDecisionToTreat = int(df[colNameTreat].dropna().median())
      df[colNameTreat].fillna(medianDecisionToTreat, inplace = True)

      existingDelay = (df[colNameDateReg] - df[colNameTreat]).dropna()
      medianDelay = int(existingDelay.median()) # delay between the decision to treat and the start of regimen

      df[colNameDateReg].loc[df[colNameDateReg].isna()] = df[colNameTreat] + medianDelay
      return df
    
    df = decToTreat(df, which = "FIRST")
    df = decToTreat(df, which = "LAST")
    return df

mainC = Main()
mainC.main("VITALSTATUS")
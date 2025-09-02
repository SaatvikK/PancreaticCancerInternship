# Branch Explanation
## 29/07_patch
See 29/07/2025 meeting notes.

## BadRecallInvestigation
See 05/08/2025 meeting
Recall for all models is < 0.1. 

Plotted KDE for all features.


- Determined it wasn't a suitable metric (see `../results/kde plots 18082025/README.md`)
Used the feature ranking of the models:
- LRC: [ 1, 12,  6, 18, 14, 17,  4,  5,  3,  2, 13, 19,  8, 10,  9,  7, 15, 11, 20, 16]
- SVM: [ 1, 13,  5, 19, 12, 17,  6,  4,  3,  2, 14, 20,  8, 10,  9,  7, 18, 11, 16, 15]
- AVG: [ 1.0, 12.5,  5.5, 18.5, 13.0, 17.0,  5.0,  4.5,  3.0,  2.0, 13.5, 19.5,  8.0, 10.0,  9.0,  7.0, 16.5, 11.0, 18.0, 15.5]

Let's remove the last common 5 features: "NUM_TUMOURS",  "LINKNUMBER", "GLEASON_PRIMARY", "AVE_WEIGHT", "GRADE"
-> this didnt help at  all

Let's remove the next last common 5 features: "GLEASON_SECONDARY", "LAST_START_DATE", "AVE_HEIGHT", "ETHNICITY", "AGE"
-> also did not help

maybe it's to do with the type of model being used? let's try a DTC and RFC.
-> They both yield a class 1 recall of ~35% (a huge improvement to the LRC/SVM value)
-> but still not very good.

Tried debiasing with ADASYN too - did nothing.


**CONCLUSION:** linear models like LRC/SVM aren't a good fit for this data. KNN, DTC, RFC are better.

## Main: 26/08/2025
- Added XGBoost
- Added Naive Bayes
- Reasoning: Non-line based models work better (ref: see KNN, RFC, and DTC results vs. SVM & LogRC)


## Calibration & SHAP (26/09/2025 - 02/09/2025)

### Calibration, Normalisation, and One Hot Encoding
Increased class 1 recall from 0.4 to 0.5 and even 0.65  (calibration) in some cases.
Used it to find the model configuration that gives the highest recall for a set precision threshold.
During this, overall accuracy remained **consistent** - which is good. Managed to increase recall without affecting overall performance negatively.

### SHAP
because one hot encoding outputs a CSR matrix, SHAP kept crashing.
Taking a small background and then densifying it into a 2D numpy array fixes this.

SHAP results: 
`shap_summary.png` - beeswarm plot: “across many test samples, which features matter most, and how?” (global view).
`shap_waterfall_sample0.png` - waterfall plot: “for this one test sample, which features drove the prediction?” (local view).

note that:
SHAP on one-hot encoded data can mess up the graphs because there's a lot of redundant features being measured which dont actually contribute much
on their own.

In `/results/shap results 02092025/` there's 2 types of results: with OHE and without OHE.

We will explain **NO** OHE results here, as it makes more sense.

From the beeswarm plot, we can see that features 13, 12, and 11 impact XGBoost the most. Whereas features 6, 4, and 3 have neglible impact.

From the waterfall plot, it can be seen that, for the random test sample, features 13, 12, 11 contributed most to the final prediction (all 3 pushed the result down significantly)






(feature order)
index: feature name
0  → ETHNICITY
1  → VITALSTATUS
2  → AVE_HEIGHT
3  → AVE_WEIGHT
4  → INTENT_OF_TREATMENT
5  → FIRST_DECISION_TO_TREAT
6  → LAST_DECISION_TO_TREAT
7  → FIRST_START_DATE
8  → LAST_START_DATE
9  → NUM_REGIMENS
10 → NUM_TUMOURS
11 → MAX_T_BEST
12 → MAX_N_BEST
13 → MAX_M_BEST
14 → MAX_STAGE_BEST
15 → GRADE
16 → AGE
17 → GLEASON_PRIMARY
18 → GLEASON_SECONDARY
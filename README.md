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
















(feature order)
index: feature name
0: PATIENTID
1: ETHNICITY
2: VITALSTATUS
3: LINKNUMBER
4: AVE_HEIGHT
5: AVE_WEIGHT
6: INTENT_OF_TREATMENT
7: FIRST_DECISION_TO_TREAT
8: LAST_DECISION_TO_TREAT
9: FIRST_START_DATE
10: LAST_START_DATE
11: NUM_REGIMENS
12: NUM_TUMOURS
13: MAX_T_BEST
14: MAX_N_BEST
15: MAX_M_BEST
16: MAX_STAGE_BEST
17: GRADE
18: AGE
19: GLEASON_PRIMARY
20: GLEASON_SECONDARY
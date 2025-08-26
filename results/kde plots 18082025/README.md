## Explanation
These KDE plots compare each features with the two class labels (0 - alive, 1 - dead).

They were taken using `../../src/kdeplot.py` on 18/08/2025, in response to the 05/08/2025 meeting where Kabita and I discussed the extremely low (< 0.1) recall of all three models for the dead (1) class.

### Features to remove
The following are features that should be removed due to their KDE plots being very similar for each class. This means it does not contribute much discrepancy-information to the model:
- AGE
- AVE_HEIGHT
- AVE_WEIGHT
- LINKNUMBER (??)

On `07/07/2025` the following features were chosen:
"PATIENTID",
"ETHNICITY",
"VITALSTATUS",
"LINKNUMBER",
"AVE_HEIGHT",
"AVE_WEIGHT",
"INTENT_OF_TREATMENT",
"FIRST_DECISION_TO_TREAT",
"LAST_DECISION_TO_TREAT",
"FIRST_START_DATE",
"LAST_START_DATE",
"NUM_REGIMENS",
"NUM_TUMOURS",
"MAX_T_BEST",
"MAX_N_BEST",
"MAX_M_BEST",
"MAX_STAGE_BEST",
"GRADE",
"AGE",
"GLEASON_PRIMARY",
"GLEASON_SECONDARY"

After this, the AGE, AVE_HEIGHT, AVE_WEIGHT, and LINKNUMBER features should be removed.

After removing these, recall for class 0 for KNN went 0.09 -> 0.1, for LRC went 0.00 -> 1, for SVM went 0.00 -> 1.
BUT the LRC and SVM accuracies plummetted down to 21% and the recall for the alive class went down to basically 0.

=> determined that KDE plots weren't sufficient for feature selection.
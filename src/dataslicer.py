# ============= Info ============= #
# Created 23/06/2025
# Last updated 23/06/2025

# For isolating pancreatic tumours (C25x) in the sim_av_tumour.csv table by ICD-10/O2 3char.
# ================================ #
import pandas as pd

tumours = pd.read_csv("../data_v2/Data/sim_av_tumour.csv")

pancreatic = tumours[tumours.SITE_ICD10_O2_3CHAR == "C25"]

pancreatic.to_csv(index=False, path_or_buf = "panc.csv")
import pandas as pd
import os

# Load your CSVs
base_dir = "C:/Users/20202932/8STAGE/data_4sites"

NYU_1 = pd.read_csv(os.path.join(base_dir, "ABIDEII-NYU_1.csv"))
NYU_2 = pd.read_csv(os.path.join(base_dir, "ABIDEII-NYU_2.csv"))
SDSU = pd.read_csv(os.path.join(base_dir, "ABIDEII-SDSU_1.csv"))
TCD  = pd.read_csv(os.path.join(base_dir, "ABIDEII-TCD_1.csv"))

pheno = pd.concat([NYU_1, NYU_2, SDSU, TCD], ignore_index=True)
pheno.columns = pheno.columns.str.strip()

# Merge the two tables:
pheno = pheno[['SUB_ID', 'SITE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX']]

pheno.to_csv(os.path.join(base_dir, "pheno_dataset.csv"), index=False)
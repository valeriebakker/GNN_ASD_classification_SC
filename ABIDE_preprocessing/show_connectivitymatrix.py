### VISUALIZE CONNECTIVITY MATRIX ###

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the matrix (streamline count or FA mean)
conn = np.loadtxt('C:/Users/20202932/8STAGE/data_4sites/ABIDEII-NYU_1/29177/session_1/dti_1/conn_FAmean.csv', delimiter=',')

# Normalize to amount of streamlines and regions
conn_norm = conn / 1000000
conn_norm = conn_norm / 120

# Make matrix symmetric
conn_norm = conn_norm + conn_norm.T - np.diag(np.diag(conn_norm))
conn = conn + conn.T - np.diag(np.diag(conn))

# Plot the single-subject FA connectivity matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conn, cmap='jet', vmin=0, vmax=0.7, square=True,
            cbar_kws={'label': 'Mean FA (connection strength)', 'format': '%.0e'})
# sns.heatmap(conn_norm, cmap='jet', square=True,
#             cbar_kws={'label': 'Normalized streamline count', 'format': '%.0e'})
plt.title('Connectivity matrix single subject')
plt.xlabel('Region index')
plt.ylabel('Region index')
plt.tight_layout()
plt.show()




### Load FA connectivity matrices for all subjects across multiple ABIDE-II sites ###

base_dir = r"C:\Users\20202932\8STAGE\data_4sites"
sites = ["ABIDEII-NYU_1", "ABIDEII-NYU_2", "ABIDEII-SDSU_1", "ABIDEII-TCD_1"]

all_conns = []
n_failed = 0

for site in sites:
    site_path = os.path.join(base_dir, site)
    
    # Loop through subjects in this site
    for subj in os.listdir(site_path):
        subj_dir = os.path.join(site_path, subj, "session_1", "dti_1")
        conn_path = os.path.join(subj_dir, "conn_FAmean.csv")

        if os.path.exists(conn_path):
            try:
                conn = np.loadtxt(conn_path, delimiter=",")
                # Make symmetric
                conn = conn + conn.T - np.diag(np.diag(conn))
                all_conns.append(conn)
            except Exception as e:
                print(f"Could not load {conn_path}: {e}")
                n_failed += 1

print(f"Loaded {len(all_conns)} matrices successfully. Failed: {n_failed}")

# Compute the group-average connectivity matrix
mean_conn = np.mean(all_conns, axis=0)
np.savetxt(r"C:\Users\20202932\8STAGE\data_4sites\mean_conn_FA.csv", mean_conn, delimiter=",")

# Plot the group-average connectivity matrix
plt.figure(figsize=(7, 7))
sns.heatmap(mean_conn, cmap="jet", vmin=0, square=True,
            cbar_kws={"label": "Mean FA (connection strength)", "shrink": 0.7})
plt.title("Average Connectivity Matrix (Mean FA across subjects)", fontsize=12)
plt.xlabel("Region index")
plt.ylabel("Region index")
plt.tight_layout()
plt.show()

n = mean_conn.shape[0]
triu_indices = np.triu_indices(n, k=1)
values = mean_conn[triu_indices]

# Get top 10 indices of strongest connections
top10_idx = np.argsort(values)[-10:][::-1]

# Convert back to region pairs (1–120)
for idx in top10_idx:
    i = triu_indices[0][idx] + 1
    j = triu_indices[1][idx] + 1
    print(f"Connection: Region {i} ↔ Region {j} | Mean FA = {values[idx]:.4f}")



# why_dropped.py
#
# Run after a simulation, from the folder that has detailed_drop_log.csv:
#     python why_dropped.py
#
# Coverage is already proven sufficient (see beam_feasibility.py), so any
# dropped user dropped for a DOWNSTREAM reason. This reads the log your
# pipeline already writes and tells you exactly which reason, and how much
# demand each reason is costing you.

import sys
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "detailed_drop_log.csv"

df = pd.read_csv(CSV)
print(f"\nLoaded {len(df):,} user-timestep rows from {CSV}\n")

# --- Outcome breakdown ----------------------------------------------------
print("FINAL STATE (all active users):")
print(df["Final_State"].value_counts().to_string())
print()

dropped = df[df["Final_State"] == "DROPPED"].copy()
if dropped.empty:
    print("No dropped users. Everyone was served by TN or NTN.")
    sys.exit(0)

print(f"DROPPED rows: {len(dropped):,}  "
      f"(demand at risk: {dropped['Demand_Mbps'].sum():,.1f} Mbps)\n")

# --- Why did NTN fail for dropped users? ----------------------------------
print("NTN_Reason for DROPPED users (the satellite-side cause):")
ntn = dropped["NTN_Reason"].fillna("N/A").value_counts()
for reason, n in ntn.items():
    mb = dropped.loc[dropped["NTN_Reason"].fillna("N/A") == reason, "Demand_Mbps"].sum()
    print(f"  {n:>6}  {mb:>9.1f} Mbps   {reason}")
print()

# --- And what had TN said about them? -------------------------------------
print("TN_Reason for DROPPED users (why 5G didn't keep them):")
print(dropped["TN_Reason"].fillna("N/A").value_counts().head(10).to_string())
print()

# --- Bucketize into the three downstream causes ---------------------------
def bucket(r):
    r = str(r)
    if "SINR too low" in r:                 return "1) SINR floor (sinr_min_db)"
    if "Congested" in r or "Empty" in r:    return "2) Beam out of bandwidth"
    if "QoS" in r:                          return "3) QoS minimum not met"
    if "No Satellite" in r:                 return "0) No satellite (geometry)"
    return "other / not evaluated"

dropped["cause"] = dropped["NTN_Reason"].map(bucket)
print("ROOT-CAUSE SUMMARY (dropped users):")
summary = dropped.groupby("cause").agg(
    users=("User_ID", "count"),
    demand_mbps=("Demand_Mbps", "sum"),
).sort_values("demand_mbps", ascending=False)
print(summary.to_string())
print()

# --- Hot hexes: where is demand piling onto one beam? ---------------------
# Approx hex by rounding coords; shows if drops cluster (-> one-beam-per-hex limit)
dropped["cell"] = (dropped["Lat"].round(1).astype(str) + "," +
                   dropped["Lon"].round(1).astype(str))
hot = (dropped.groupby("cell")
       .agg(dropped_rows=("User_ID", "count"), demand=("Demand_Mbps", "sum"))
       .sort_values("demand", ascending=False).head(10))
print("TOP 10 LOCATIONS BY DROPPED DEMAND (clustering => one-beam-per-hex limit):")
print(hot.to_string())
print()

print("WHAT TO DO:")
print(" * If cause #2 dominates: a single hex's demand exceeds one beam's")
print("   capacity. Your allocator fires ONE beam per hex (beam_allocator.py")
print("   ~line 1808) while the satellite still has free beams. Allow >1 beam")
print("   per hex, or raise bandwidth_hz.")
print(" * If cause #1 dominates: raise EIRP/G_T, or lower sinr_min_db, or")
print("   check the antenna roll-off (theta_3db_deg) penalty.")
print(" * If cause #3 dominates: qos_min_mbps is too high for leftover bandwidth.")
print(" * If cause #0 appears at all: a skyfield-vs-geometry edge mismatch or")
print("   a beam-slot race; rare given your feasibility result.")
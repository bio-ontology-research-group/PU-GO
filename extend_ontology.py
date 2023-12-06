import sys
import pandas as pd
import mowl
mowl.init_jvm("10g")


ont = sys.argv[1]
if not ont in ["mf", "bp", "cc"]:
    raise ValueError("Ontology must be one of 'mf', 'bp', 'cc'")

df = pd.read_pickle(f"../data-sim/{ont}/train_data.pkl")





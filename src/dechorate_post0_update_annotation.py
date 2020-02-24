import numpy as np
import pandas as pd

# load csv with objects position
path_to_positioning_note = './data/dECHORATE/annotations/dECHORATE_positioning_note.csv'
pos_note_df = pd.read_csv(path_to_positioning_note)
# load csv with recordings annotation
path_to_recordings_note = './data/dECHORATE/annotations/dECHORATE_recordings_note.csv'
rec_note_df = pd.read_csv(path_to_recordings_note)

print(pos_note_df)
print(rec_note_df)

for row in rec_note_df.import
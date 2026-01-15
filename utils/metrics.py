import pandas as pd
import pathlib as Path

path = Path.Path("/Users/khaledalamro/Desktop/NLP_Project/Project/data/arabic_dataset_classifiction.csv")
df = pd.read_csv(path)
print(df[df["target"] == 0].head(10), "\n#####################")
print(df[df["target"] == 1].head(10), "\n#####################")
print(df[df["target"] == 2].head(10), "\n#####################")
print(df[df["target"] == 3].head(10), "\n#####################")
print(df[df["target"] == 4].head(10), "\n#####################")

topic_map = {
    0: 'Culture',
    1: 'Diverse',
    2: 'Economy',
    3: 'Politic',
    4: 'Sport'
}
import pandas as pd

df = pd.read_json("processed.json")
df.drop(columns=['DRUG_CID', "DRUG_NAME"], inplace=True)
df['INDEX'] = df.index
df1 = df.copy()
df2 = df.copy()
df1.columns = ['DRUG_ID1', 'INTERACTIONS1', 'STRUCTURE1', 'TARGET1', 'ENZYME1', 'PATH1', 'SIDE_EFFECTS1', 'INDICATIONS1', 'OFFSIDE_EFFECT1', 'INDEX1']
df2.columns = ['DRUG_ID2', 'INTERACTIONS2', 'STRUCTURE2', 'TARGET2', 'ENZYME2', 'PATH2', 'SIDE_EFFECTS2', 'INDICATIONS2', 'OFFSIDE_EFFECT2', 'INDEX2']
df = pd.merge(df1, df2, how="cross")
df = df[df['DRUG_ID1'] != df['DRUG_ID2']]
df.reset_index(inplace=True)
df['INTERACTION'] = df.apply(lambda row: row['INTERACTIONS1'][row['INDEX2']], axis=1)

df['id_pair'] = df[['DRUG_ID1', 'DRUG_ID2']].apply(lambda x: '-'.join(sorted(map(str, x))), axis=1)
df.drop_duplicates(subset='id_pair', inplace=True)
df.dropna(inplace=True)
df.drop(columns=["index", 'INTERACTIONS1', 'INTERACTIONS2', 'INDEX1', 'INDEX2', 'id_pair'], inplace=True)

to_edit = ['STRUCTURE', 'TARGET', 'ENZYME', 'PATH', 'SIDE_EFFECTS', 'INDICATIONS', 'OFFSIDE_EFFECT']
for column in to_edit:
    column1 = column + '1'
    column2 = column + '2'
    df[column + '_JAC_SIM'] = df.apply(lambda row: len(set(row[column1]).intersection(set(row[column2]))) /
                                                   len(set(row[column1]).union(set(row[column2]))), axis=1)
    df.drop(columns=[column1, column2], inplace=True)

df.reset_index(inplace=True)
df.drop(columns=["index"], inplace=True)
df.to_json("jaccard.json")
print(df.head().to_string())

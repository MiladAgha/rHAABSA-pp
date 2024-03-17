import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np

def json_to_df(dataset):
    input_file = 'data/ARTSData/{}.json' .format(dataset)

    with open(input_file, 'r', encoding='utf-8') as fr:
        data = json.load(fr)

    df = pd.DataFrame(data)
    df = df.transpose()
    df['sid'] = df['id'].apply(lambda x: x[:-1] if x[-1:].isdigit() else x).apply(lambda x: x[:-1] if x[-1:].isdigit() else x).apply(lambda x: x[:-1] if x[-1:] == '_' else x)
    df.drop(['id','from','to','term'], axis=1, inplace=True)
    return df

def rf_sentence_label(dataset):
    df = json_to_df(dataset)
    df = pd.DataFrame(df.polarity.value_counts().reset_index().values, columns=['polarity', 'count']).set_index('polarity')
    df[dataset] = df['count']/df['count'].sum()
    return df

def rf_sentence_labels():
    df = rf_sentence_label('2015train')
    df.drop(['count'], axis=1, inplace=True)
    df['2015test'] = rf_sentence_label('2015test')['2015test']
    df['ont2015test'] = rf_sentence_label('ont2015test')['ont2015test']
    df['2016train'] = rf_sentence_label('2016train')['2016train']
    df['2016test'] = rf_sentence_label('2016test')['2016test']
    df['ont2016test'] = rf_sentence_label('ont2016test')['ont2016test']
    return df

def distribution_aspects_per_sentence(dataset):
    df = json_to_df(dataset)
    df = pd.DataFrame(df.sid.value_counts().reset_index().values, columns=["sid", "counts"]).set_index('sid')
    df = json_to_df(dataset).merge(df, on='sid')
    df = pd.DataFrame(df.counts.value_counts().reset_index().values, columns=["nasp", dataset]).set_index('nasp')
    df[dataset] = df[dataset]/df[dataset].sum()
    df_new = df.iloc[:3].copy()
    sum_row = df.iloc[3:].sum()
    sum_row.name = 4
    df_new = df_new.append(sum_row)
    return df_new

def distribution_aspects_per_sentence_ont(dataset):
    df = json_to_df(dataset)
    df = pd.DataFrame(df.sid.value_counts().reset_index().values, columns=["sid", "counts"]).set_index('sid')   
    df = json_to_df('ont'+dataset).merge(df, on='sid')
    df = pd.DataFrame(df.counts.value_counts().reset_index().values, columns=["nasp", 'ont'+dataset]).set_index('nasp')
    df['ont'+dataset] = df['ont'+dataset]/df['ont'+dataset].sum()
    df_new = df.iloc[:3].copy()
    sum_row = df.iloc[3:].sum()
    sum_row.name = 4
    df_new = df_new.append(sum_row)
    return df_new

def distribution_aspects_per_sentence_full():
    df = pd.concat([distribution_aspects_per_sentence('2015train'), distribution_aspects_per_sentence('2015test')], axis=1)
    df = pd.concat([df, distribution_aspects_per_sentence_ont('2015test')], axis=1)
    df = pd.concat([df, distribution_aspects_per_sentence('2016train')], axis=1)
    df = pd.concat([df, distribution_aspects_per_sentence('2016test')], axis=1)
    df = pd.concat([df, distribution_aspects_per_sentence_ont('2016test')], axis=1)
    return df

def plot_rf_sentence_labels():

    df = rf_sentence_labels()
    df = df.sort_values('polarity')
    df = df[['2015train','2015test','2016train','2016test']].transpose()

    fig, ax = plt.subplots()

    ax.bar(df.index, np.multiply(df['positive'],100), label = "positive", color = "#2C858D")
    ax.bar(df.index, np.multiply(df['neutral'],100), bottom = np.multiply(df['positive'],100), label = "neutral", color = "#74CEB7")
    ax.bar(df.index, np.multiply(df['negative'],100), bottom = np.add(np.multiply(df['positive'],100), np.multiply(df['neutral'],100)), label = "negative", color = "#C9FFD5")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.yaxis.set_minor_locator(tck.MultipleLocator(10))
    ax.set_xticks([0.5, 2.5], minor=True)
    ax.set_xticklabels(['\nTraining','\nTest','\nTraining','\nTest'])
    ax.set_xticklabels(['2015 Dataset','2016 Dataset'], fontsize=11, fontweight='bold', minor=True)
    ax.xaxis.set_ticks_position('none') 

    plt.ylabel('Aspects in data (in %)')
    plt.grid(axis='y',which='major', linestyle='-.', linewidth='0.5', color='dimgray')
    plt.grid(axis='y',which='minor', linestyle='-.', linewidth='0.5', color='dimgray')
    plt.show()

def plot_distribution_aspects_per_sentence_full():

    df = distribution_aspects_per_sentence_full()
    df = df[['2015train','2015test','2016train','2016test']].transpose()

    fig, ax = plt.subplots()

    ax.bar(df.index, np.multiply(df[1],100), label = "1", color = "#004056")
    ax.bar(df.index, np.multiply(df[2],100), bottom = np.multiply(df[1],100), label = "2", color = "#2C858D")
    ax.bar(df.index, np.multiply(df[3],100), bottom = np.add(np.multiply(df[1],100), np.multiply(df[2],100)), label = "3", color = "#74CEB7")
    ax.bar(df.index, np.multiply(df[4],100), bottom = np.add(np.add(np.multiply(df[1],100), np.multiply(df[2],100)), np.multiply(df[3],100)), label = ">=4", color = "#C9FFD5")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    ax.yaxis.set_minor_locator(tck.MultipleLocator(10))
    ax.set_xticks([0.5, 2.5], minor=True)
    ax.set_xticklabels(['\nTraining','\nTest','\nTraining','\nTest'])
    ax.set_xticklabels(['2015 Dataset','2016 Dataset'], fontsize=11, fontweight='bold', minor=True)
    ax.xaxis.set_ticks_position('none') 

    plt.ylabel('Sentences in data (in %)')
    plt.grid(axis='y',which='major', linestyle='-.', linewidth='0.5', color='dimgray')
    plt.grid(axis='y',which='minor', linestyle='-.', linewidth='0.5', color='dimgray')
    plt.show()
    
# print(distribution_aspects_per_sentence_full())
# print(rf_sentence_labels())

# plot_rf_sentence_labels()
# plot_distribution_aspects_per_sentence_full()

# 2014, testsize: 1120, remainingsize: 503, acc_ont: 0.7569
# 2015, testsize: 559, remainingsize: 283, acc_ont: 0.8007
# 2016, testsize: 623, remainingsize: 239, acc_ont: 0.8620    

def calculate_ARS(year, set):
    input_file = 'data/ARTSData/ARTS{}test.json' .format(year)
    key_file = 'data/ARTSData/ont{}.json' .format(year)
    result_file = 'data/results/{}{}.txt' .format(year, set)

    with open(input_file, 'r', encoding='utf-8') as fr:
        data = json.load(fr)

    with open(key_file, 'r', encoding='utf-8') as fr:
        ont_keys = json.load(fr)

    df = pd.DataFrame(data).transpose()

    df['sid'] = df['id'].str.replace(r'_adv[1-3]', '', regex=True)
    df['adv'] = df['id'].str.extract(r'_adv([1-3])').fillna(0).astype(int)
    df.drop(['id','from','to','term', 'sentence'], axis=1, inplace=True)

    df['c'] = pd.read_csv(result_file, sep=" ", header=None).set_index(df.index)

    acc = sum(df['c'])/len(df['c'])
    df_ARS = pd.DataFrame(df[['sid','c']].groupby('sid').sum())
    ARS = len(df_ARS.loc[df_ARS['c'] == 0])/len(df_ARS['c'])

    df = df[df['adv'] == 0]
    acc_0 = sum(df['c'])/len(df['c'])

    res = pd.DataFrame(
        {
            'acc': [acc],
            'acc_0': [acc_0],
            'ARS': [ARS]
        },
        index = ['{}_{}' .format(year,set)])
    return res

res = pd.concat([calculate_ARS(2015, 'BERT'),calculate_ARS(2016, 'BERT')]).transpose()
print(res)
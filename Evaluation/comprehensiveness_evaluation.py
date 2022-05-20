# Created by Huyen Nguyen at 5/20/2022

# preprocess the concepts
# Defining pre-processing function

from nltk.stem import WordNetLemmatizer


def preprocessEntity(text):
    processed1 = lowercase(text)
    processed2 = lemmatization(processed1)
    normalized = normalizeCOVID(processed2)
    return normalized


def lowercase(text):
    lowercased = text.lower()
    return lowercased


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    return lemmatized


def normalizeCOVID(text):
    covid_variants1 = ['coronarivus disease', 'sarsr-cov', '2019ncov', '2019 ncov',
                       'severe acute respiratory syndrome-related coronavirus 2', 'sars-cov2', 'wuhan virus',
                       'covid-19 (covid-19)', 'covid19 coronavirus', 'coronarivus', 'sars-cov-2', '2019-ncov',
                       'covid 19', 'covid19', 'wuhan coronavirus', 'chinese coronavirus', 'covidー19',
                       'novel coronavirus']
    covid_variants2 = ['corona', 'sars-cov']
    for variant in covid_variants1:
        if variant in text:
            text = text.replace(variant, 'covid-19')

    if (('corona' in text) and ('coronavirus' not in text)):
        text = text.replace('corona', 'covid-19')

    if (('sars-cov' in text) and ('sars-cov-2' not in text)):
        text = text.replace('corona', 'covid-19')
    return text


# preprocessEntity('normalized bananas coronarivus disease coronarivus disease sars-cov-2')

import pandas as pd

paths = [
    r"C:\Users\huyen\OneDrive - UNT System\PROJECTS\COVID19_paper\KG paper\Evaluation\Evaluation_Students\comprehensiveness\all.csv",
    r"C:\Users\huyen\OneDrive - UNT System\PROJECTS\COVID19_paper\KG paper\Evaluation\Evaluation_Students\comprehensiveness\COVID-19.csv",
    r"C:\Users\huyen\OneDrive - UNT System\PROJECTS\COVID19_paper\KG paper\Evaluation\Evaluation_Students\comprehensiveness\SARS-CoV-2.csv"]

all_entities = []
for path in paths:
    with open(path, 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)
    data = data.fillna('')
    data = data.applymap(preprocessEntity)
    data = data.apply(lambda x: x.replace(x, 'covid-19 outbreak') if 'outbreak' in x else x)
    if len(data.columns) < 2:
        data['head'] = data['head'].apply(lambda x: x.replace(x, 'covid-19 outbreak') if 'outbreak' in x else x)
        data['head'] = data['head'].apply(lambda x: x.replace(x, 'covid-19') if 'wuhan' in x else x)
        all_entities.extend(data['head'].values.tolist())
        # print('1111-----', len(all_entities))
    else:
        for col in data.columns:
            data[col] = data[col].apply(lambda x: x.replace(x, 'covid-19') if 'wuhan' in x else x)
            # print('2222---------', len(data[col].values.tolist()))
            all_entities.extend(data[col].values.tolist())

covid_variants = ['covid2019', 'covid-2019', 'covid', 'coronavirus disease 2019', 'covid-19 acute respiratory disease',
                  'covid-19 virus', 'covid-19', '2019 covid-19 respiratory syndrome',
                  'acute respiratory syndrome coronavirus 2', 'covid-19 (covid-19)']

pandemic_variants = ['2019–2021 coronavirus pandemic', 'covid-19 pandemic', 'the great disaster of 2020']
final = set([item for item in all_entities if type(item) is str])
final = [i for i in final if len(i) > 1]
final = [i.replace('[covid-19]', '') for i in final]
print('all scrapped concepts from Wikidata: ', len(final))
final

# Comprehensiveness over 10 folds
# Load the 10 fold data
import glob

root_p = r"C:\Users\huyen\OneDrive\Documents\GitHub\CORD-19-KG\Data\all-final-cleaned-triple3-10sets"
filter_variants = [i for i in final if i not in covid_variants + pandemic_variants]
n_folders = 10
comprehensive_score = {}

with open(r'C:\Users\huyen\OneDrive\Documents\GitHub\CORD-19-KG\Evaluation\result\comprehensiveness_scores.csv',
          'w') as file:
    file.write('subset,compre_score\n')

for i in range(1, n_folders + 1):
    all_ents = []
    wikidata_mapped = []
    not_mapped = []
    fold_p = r"C:\Users\huyen\OneDrive\Documents\GitHub\CORD-19-KG\Data\all-final-cleaned-triple3-10sets\subset_%s.csv" % i
    with open(fold_p, 'r', encoding='utf-8') as f:
        data = pd.read_csv(fold_p)
    #     print(data.columns)
    all_ents.extend(data['subject'].values.tolist())
    all_ents.extend(data['object'].values.tolist())
    #     print('all entities with duplicates: %s'%len(all_ents))
    all_ents_set = set(all_ents)
    #     print('all entities with NO duplicates: %s'%len(all_ents_set))

    for concept in set(final):
        if concept in all_ents_set:
            wikidata_mapped.append(concept)
        else:
            not_mapped.append(concept)
    compre_score = len(set(wikidata_mapped)) / (len(filter_variants) + 2)
    print('subset {}: {}'.format(i, compre_score))
    with open(r'C:\Users\huyen\OneDrive\Documents\GitHub\CORD-19-KG\Evaluation\result\comprehensiveness_scores.csv',
              'a') as file:
        file.write('subset_{},{}\n'.format(i, compre_score))
    comprehensive_score['subset_%s' % i] = compre_score


# comprehensive_score
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,6))
y = [score for score in comprehensive_score.values()]
x = [str(i) for i in range(1,11)]
plt.plot(x,y, 'v--', color= 'darkorange' )#, label= query )

plt.xlabel('subsets')
plt.ylabel('comprehensiveness scores')
plt.legend(loc='upper left', ncol = 1)
# plt.title( 'comprehensive scores over 10 subsets')
plt.show()
fig.savefig(r"C:\Users\huyen\OneDrive\Documents\GitHub\CORD-19-KG\Evaluation\result\KG_eval\comprehensiveness_10folds.png" )
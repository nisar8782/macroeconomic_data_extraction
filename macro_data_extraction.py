
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from IPython.display import display
import requests
import math
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons





# ------Macro udf

def generate_info_eu_efta_countries():
    # Definisci le liste di paesi
    eu_countries = ['Austria', 'Belgium','Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']
    efta_countries = ['Iceland', 'Norway','Liechtenstein', 'Switzerland']
    uk = ['United Kingdom']
    eu_cand_countries = ['Bosnia and Herzegovina', 'Montenegro','Moldova', 'North Macedonia', 'Albania', 'Serbia', 'Türkiye', 'Ukraine']

    # Crea i dataframes
    df1 = pd.DataFrame(eu_countries, columns=['country'])
    df2 = pd.DataFrame(efta_countries, columns=['country'])
    df3 = pd.DataFrame(uk, columns=['country'])
    df4 = pd.DataFrame(eu_cand_countries, columns=['country'])

    # Aggiungi la colonna 'list' a ciascun dataframe
    df1['list'] = 'eu_countries'
    df2['list'] = 'efta_countries'
    df3['list'] = 'uk'
    df4['list'] = 'eu_cand_countries'

    # Unisci i dataframes
    info_eu_efta_countries = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    return info_eu_efta_countries


def check_countries(EU_countries, info_eu_efta_countries):
    # Filtra il dataframe con i paesi dell'EU
    df_eu = info_eu_efta_countries[info_eu_efta_countries['country'].isin(EU_countries)]
    
    # Trova i paesi che non appartengono a 'eu_countries'
    non_eu_countries = df_eu[df_eu['list'] != 'eu_countries']['country'].tolist()

    # Verifica se esistono paesi non EU
    if non_eu_countries:
        print("You also selected extra eu countries (es. EFTA)")
        print("NOTE: these data could have missing or be not accurate. So, check them.")
        non_eu_countries = pd.DataFrame(non_eu_countries, columns=['no EU coutries'])
        return non_eu_countries
    
def test_single_df(codes, years, eurostat_dictionary, EU_countries, name_index, third_col_name):
    # path_save_file = path_save_file
    desired_years = years

    existing_df = None  # Inizializza il dataframe iniziale come None

    # for i, code in enumerate(codes, eurostat_dictionary):
    for i, code in enumerate(codes):
        # df = eurostat.get_data_df(code)
        resp = requests.get(f'https://ec.europa.eu/eurostat/databrowser-backend/api/extraction/1.0/LIVE/false/json/en/{code}?cacheId=1682067600000-4.5.15%2520-%25202023-05-23%252008%253A46')
        # print(resp)
        sorted_dict = dict(sorted(resp.json()['value'].items(), key=lambda x: int(x[0])))
        year_list = []
        # getting all of the years and appending into the year list
        for key_time,value_time  in resp.json()['dimension']['time']['category']['label'].items():
            year_list.append(key_time)
        geo=[]
        for key_geo,value_geo in resp.json()['dimension']['geo']['category']['index'].items():
            geo.append(key_geo)
        for m in range(0, len(year_list)*len(geo)):
            key = str(m)
            if key not in sorted_dict:
                sorted_dict[key] = '0'
        sorted_dict = dict(sorted(sorted_dict.items(), key=lambda x: int(x[0])))
        values = []
        rows = []
        for key, value in sorted_dict.items():
            values.append(value)
        values_array = np.array(values[0:len(year_list)*len(geo)], dtype=float).reshape(len(geo),len(year_list))
        # Create a DataFrame from values_array
        df = pd.DataFrame(values_array, columns=year_list, index=geo)
        df = df.reset_index().rename(columns={'index': 'geo\TIME_PERIOD'})
        # EDIT ROW STARTING TAB
        # rename 'geo' column
        df.rename({'geo\TIME_PERIOD': 'geo'}, inplace=True, axis=1)
        # create country column
        df['country'] = df['geo'].replace(eurostat_dictionary)
        # drop column from geo (included)
        df = df.drop(df.columns[:df.columns.get_loc('geo') + 1], axis=1)
        # select country available in EU_countries list (above)
        df = df[df['country'].isin(EU_countries)]
        # drop country just imported
        df.drop_duplicates(subset='country', inplace=True)
        # select colum country and year (eg. 2011, ...)
        df = df.loc[:, ['country'] + desired_years]
        # melt df to create column: country (reference), year, categories (eg. GDP, TRADE_B)
        df = pd.melt(df, id_vars=['country'], var_name="year", value_name="Value")        
        # convert data to float
        for col in df.columns[df.columns.get_loc('year') + 1:]:
            df[col] = df[col].astype('float64') 
        
        # Rename by third column in each cycle imported with mapping
#         third_col_name = column_custom_mapping.get(code, 'GDP')
        df.rename(columns={"Value": name_index}, inplace=True)
        
        # MERGE column country and year with new column (reference column: country, year)
        if i == 0:
            existing_df = df  # Assign first df imported as initial df
        else:
            existing_df = existing_df.merge(df[['country', 'year', third_col_name]], on=['country', 'year'], how='left')
            
    # Replace NaN values with 0  
#     existing_df.fillna(0, inplace=True)
    
    return existing_df

def update_dataframes(codes, eurostat_dictionary, years,EU_countries,column_custom_mapping):
    # path_save_file = path_save_file
    desired_years = years

    existing_df = None  # Inizializza il dataframe iniziale come None

    for i, code in enumerate(codes):
        # df = eurostat.get_data_df(code)
        resp = requests.get(f'https://ec.europa.eu/eurostat/databrowser-backend/api/extraction/1.0/LIVE/false/json/en/{code}?cacheId=1682067600000-4.5.15%2520-%25202023-05-23%252008%253A46')
        # print(resp)
        sorted_dict = dict(sorted(resp.json()['value'].items(), key=lambda x: int(x[0])))
        year_list = []
        # getting all of the years and appending into the year list
        for key_time,value_time  in resp.json()['dimension']['time']['category']['label'].items():
            year_list.append(key_time)
        geo=[]
        for key_geo,value_geo in resp.json()['dimension']['geo']['category']['index'].items():
            geo.append(key_geo)
        for m in range(0, len(year_list)*len(geo)):
            key = str(m)
            if key not in sorted_dict:
                sorted_dict[key] = '0'
        sorted_dict = dict(sorted(sorted_dict.items(), key=lambda x: int(x[0])))
        values = []
        rows = []
        for key, value in sorted_dict.items():
            values.append(value)
        values_array = np.array(values[0:len(year_list)*len(geo)], dtype=float).reshape(len(geo),len(year_list))
        # Create a DataFrame from values_array
        df = pd.DataFrame(values_array, columns=year_list, index=geo)
        df = df.reset_index().rename(columns={'index': 'geo\TIME_PERIOD'})
        # EDIT ROW STARTING TAB
        # rename 'geo' column
        df.rename({'geo\TIME_PERIOD': 'geo'}, inplace=True, axis=1)
        # create country column
        df['country'] = df['geo'].replace(eurostat_dictionary)
        # drop column from geo (included)
        df = df.drop(df.columns[:df.columns.get_loc('geo') + 1], axis=1)
        # select country available in EU_countries list (above)
        df = df[df['country'].isin(EU_countries)]
        # drop country just imported
        df.drop_duplicates(subset='country', inplace=True)
        # select colum country and year (eg. 2011, ...)
        df = df.loc[:, ['country'] + desired_years]
        # melt df to create column: country (reference), year, categories (eg. GDP, TRADE_B)
        df = pd.melt(df, id_vars=['country'], var_name="year", value_name="Value")        
        # convert data to float
        for col in df.columns[df.columns.get_loc('year') + 1:]:
            df[col] = df[col].astype('float64') 
        
        # Rename by third column in each cycle imported with mapping
        third_col_name = column_custom_mapping.get(code, 'GDP')
        df.rename(columns={"Value": third_col_name}, inplace=True)
        
        # MERGE column country and year with new column (reference column: country, year)
        if i == 0:
            existing_df = df  # Assign first df imported as initial df
        else:
            existing_df = existing_df.merge(df[['country', 'year', third_col_name]], on=['country', 'year'], how='left')
            
    # Replace NaN values with 0  
#     existing_df.fillna(0, inplace=True)
    
    return existing_df

# ------ NaN showing

def nan_sum_up(df):
    
    print('Following check also if data:')
    print('- Missing completely at random (MCAR)')
    print('- Missing at random (MAR)')
    print('- Missing not at random (MNAR)\n')
    
    # -- Get just number columns
    df_number = df.select_dtypes(include=[np.number])
    
    # -- Get data we are interasting in
    # NaN
    total_nan = df_number.isna().sum().sum()
    total_nan_col = df_number.isna().sum()
    # Zero
    total_zero = (df_number == 0).sum().sum() 
    total_zero_col = (df_number == 0).sum()    
    # Data
    total_data = df_number.size
    total_val_col = df_number.sum() 

    # -- Get data
    # NaN
    nan_total_percentage = total_nan / total_data * 100
    percentage_nan_to_col = (total_nan_col / total_val_col) * 100
    percentage_nan = (total_nan_col / total_nan) * 100
    # Zero
    zero_total_percentage = total_zero / total_data * 100
    percentage_zero_to_col = (total_zero_col / total_val_col) * 100 
    percentage_zero = (total_zero_col / total_zero) * 100
    
    
    print(f"*Total data*: {total_data}")
    print(f"*Total NaN in dataframe*: {total_nan}") 
    print(f"*Total Zero in dataframe*: {total_zero}")
    print(f"*Percentage of NaN values to total size*: {nan_total_percentage:.2f}%")
    print(f"*Percentage of Zero values to total size*: {zero_total_percentage:.2f}%")
    print(f"*Total NaN + Total Zero*: {(nan_total_percentage + zero_total_percentage):.2f}%")
    # ---
    
    print("\n*Column: NaN&Zero to values*")
    perc_nanzero_to_values_df = pd.DataFrame({'%Zero':percentage_zero_to_col, '%NaN':percentage_nan_to_col})
    perc_nanzero_to_values_df = perc_nanzero_to_values_df.applymap('{:.0f}%'.format)    
    display(perc_nanzero_to_values_df)
    print('\n')
    
    print("*Total NaN&Zero concentration*")
    percentage_nan_df = pd.DataFrame({'%Zero':percentage_zero, '%NaN':percentage_nan})
    percentage_nan_df = percentage_nan_df.applymap('{:.0f}%'.format)
    display(percentage_nan_df)
    
    print("\n*Total NaN&Zero concentration for each country. Threshold: if nan_sum[col] > 1 or zero_sum[col] > 1*")
    df_iter = df.groupby('country')

    # Initialize list to store the results
    result_nan_country = []

    for country, group in df_iter:
        no_nan_counts = group.count()
        nan_sum = group.isna().sum()
        zero_sum = (group == 0).sum()

        for col in group.columns:
            # Compute the ratio of NaN and Zero to non-NaN and non-zero values
            ratio_nan = nan_sum[col] / (nan_sum[col] + no_nan_counts[col])
            ratio_zero = zero_sum[col] / (nan_sum[col] + no_nan_counts[col])

            # Threshold
            if nan_sum[col] > 1 or zero_sum[col] > 1:
                nan_val_country = {
                    'Country': country,
                    'Column': col,
                    'Zero': zero_sum[col],
                    '(%)Zero-to-Val': 'NaN' if pd.isna(ratio_zero) else '{:.0f}%'.format(ratio_zero * 100),  # Convert ratio to percentage
                    'NaN': nan_sum[col],
                    '(%)NaN-to-Val': '{:.0f}%'.format(ratio_nan * 100)  # Convert ratio to percentage 
                }

                result_nan_country.append(nan_val_country)

    result_nan_country_df = pd.DataFrame(result_nan_country)
    result_nan_country_df = result_nan_country_df.sort_values(by='Column', ascending=True)
    display(result_nan_country_df)

    print("(High MNAR is an alarm that data are wrong)\n")
    return

def columns_zeronanover3(df, threshold=0.3):
    na_or_zero = input('Enter "0" to search for zeros or "na" to search for missing values: ')
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    total_count = {col: (df[col] == 0).sum() if na_or_zero == "0" else df[col].isna().sum() for col in numeric_columns}
    results = {col: [] for col in numeric_columns}  
    count_results = {col: 0 for col in numeric_columns}

    for country in df['country'].unique():
        df_country = df[df['country'] == country]
        for col in numeric_columns:
            if na_or_zero == "0":
                condition = (df_country[col] == 0).sum() / len(df_country[col]) > threshold
                count = (df_country[col] == 0).sum()
            elif na_or_zero == "na":
                condition = df_country[col].isna().sum() / len(df_country[col]) > threshold
                count = df_country[col].isna().sum()
            if condition:
                results[col].append(country)
                count_results[col] += count

    col_to_replace = results
    percent_results = {col: (count_results[col] / total_count[col]) * 100 if total_count[col] != 0 else 0 for col in numeric_columns}
    return col_to_replace, percent_results

def to_check_zeronan(df_original, col_to_replace, percent_results):
    to_check = input('Select col_to_replace or percent_results: ')
    print('\nCountry with more than 30% zero or nan')
#     df_selected = input('Select df to total countries count (now available: df_perc)')
#     df_map = {'df_perc': df_perc}
#     df_selected = df_map[df_selected]

    # Supponendo che `col_to_replace` e `percent_results` siano dizionari
    if to_check == "col_to_replace":

        # Trasforma le liste vuote in None
        for key in col_to_replace.keys():
            if not col_to_replace[key]:
                col_to_replace[key] = [None]         
        
        # Print countries found for single coulmn
        print('\nFocus on columns and countries')
        df = pd.DataFrame.from_dict(col_to_replace, orient='index').transpose()
        col_to_replace_no_nan = df.stack().dropna().tolist()
        return df, col_to_replace_no_nan  # print DataFrame
    
    elif to_check == "percent_results":
        print('\nColumn rate nan or zero compared to total')
        df_check_perc = pd.DataFrame(percent_results, index=[0])
        
        # Trasforma le liste vuote in None
        for key in col_to_replace.keys():
            if not col_to_replace[key]:
                col_to_replace[key] = [None]
        
        # Get percentage countries-to-total countries
        df_tot_perc = pd.DataFrame.from_dict(col_to_replace, orient='index').transpose()
        col_to_replace_no_nan = df_tot_perc.stack().dropna().tolist()
        tot_countries_found = len(col_to_replace_no_nan) / df_original['country'].nunique()
        tot_countries_found = '{:.0%}'.format(tot_countries_found)
        return df_check_perc, tot_countries_found  # print DataFrame
    
# Replace all zero values from list 'country_tot'
def replace_zero_values(df,col_to_replace_no_nan):
    
    print(col_to_replace_no_nan)
    col_to_replace_no_nan
    
    # Chiede all'utente di inserire i valori da rimuovere, separati da una virgola
    input_values = input("\nEnter the values to remove, separated by a comma: ")

    # Divide la stringa in una lista di valori
    values_to_remove = input_values.split(',')

    # Rimuove gli spazi bianchi iniziali e finali da ciascun valore
    values_to_remove = [value.strip() for value in values_to_remove]

    # Trasforma la lista in un set (facoltativo, ma può rendere l'operazione più veloce se ci sono molti valori)
    values_to_remove = set(values_to_remove)

    # Ora puoi utilizzare values_to_remove come prima
    col_to_replace_no_nan_updated = [x for x in col_to_replace_no_nan if x not in values_to_remove]

    for country in df['country'].unique():
        if country in col_to_replace_no_nan_updated:
            for col in df.select_dtypes(include=[np.number]).columns:
                df.loc[(df['country'] == country) & (df[col] == 0), col] = np.nan
                df_test = df.copy()
    return df_test

def threshold_over_dcc(df, threshold=0.3):
    # Passaggio a)
    print('ddc meaning: data, columns, countries\n')
    print('a): Are NaN over 30% of total data?')
    total_values = [df.select_dtypes(include=[np.number]).size]
    total_nan_values = [df.isna().sum().sum()]
    percent_nan_values = [total_nan_values[0] / total_values[0] * 100]
    threshold_over = ['Yes' if total_nan_values[0] / total_values[0] > threshold else 'No']
    df_a = pd.DataFrame({'Total values': total_values, 'Total NaN values': total_nan_values, '% NaN values': [f'{x:.0f}%' for x in percent_nan_values], 'Threshold 30% over': threshold_over})
    df_data = display(df_a)

    # Passaggio b)
    print('\nb) Are NaN over 30% in some columns?')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    col_names = []
    total_values_b = []
    total_nan_values_b = []
    percent_nan_values_b = []
    threshold_over_b = []
    for col in numeric_columns:
        col_names.append(col)
        total_values_col = df[col].size
        total_values_b.append(total_values_col)
        nan_values = df[col].isna().sum()
        total_nan_values_b.append(nan_values)
        percent_nan_values_b.append(nan_values / total_values_col * 100)
        threshold_over_b.append('Yes' if nan_values / total_values_col > threshold else 'No')
    df_b = pd.DataFrame({'Column name': col_names, 'Total values': total_values_b, 'Total NaN values': total_nan_values_b, '% NaN values': [f'{x:.0f}%' for x in percent_nan_values_b], 'Threshold 30% over': threshold_over_b})
    df_columns = display(df_b)

    # Passaggio c)
    print('\nc) Are NaN over 30% in some countries?')
    countries = []
    total_values_c = []
    total_nan_values_c = []
    percent_nan_values_c = []
    threshold_over_c = []
    for country in df['country'].unique():
        countries.append(country)
        country_df = df[df['country'] == country]
        total_values_country = country_df.select_dtypes(include=[np.number]).size
        total_values_c.append(total_values_country)
        total_nan_values_country = country_df.select_dtypes(include=[np.number]).isna().sum().sum()
        total_nan_values_c.append(total_nan_values_country)
        percent_nan_values_c.append(total_nan_values_country / total_values_country * 100)
        threshold_over_c.append('Yes' if total_nan_values_country / total_values_country > threshold else 'No')
    df_c = pd.DataFrame({'Country': countries, 'Total values': total_values_c, 'Total NaN values': total_nan_values_c, '% NaN values': [f'{val:.0f}%' for val in percent_nan_values_c], 'Threshold 30% over': threshold_over_c})
    df_countries = display(df_c)
    return df_data, df_columns, df_countries

def nan_over_threshold(df):
    print('Threshold: more than 30% NaN for each country (each value in df show how much nan reference contains)')

    # Seleziona solo le colonne numeriche
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Crea un DataFrame vuoto per contenere i risultati
    result = pd.DataFrame()

    # Itera su ogni colonna numerica
    for col in numeric_columns:

        # Calcola il numero totale di valori per country per la colonna corrente
        total_counts_country = df.groupby('country')[col].size()

        # Calcola il numero di valori nulli per country per la colonna corrente
        nan_counts_country = df[df[col].isna()].groupby('country')[col].size()

        # Calcola la percentuale di valori NaN
        nan_perc_country = (nan_counts_country / total_counts_country) * 100

        # Filtra i paesi con più del 30% di valori NaN
        high_nan_countries = nan_perc_country[nan_perc_country > 30]

        # Aggiungi i risultati al DataFrame dei risultati
        result = pd.concat([result, high_nan_countries.rename(col)], axis=1)

    # Verifica se il DataFrame dei risultati è vuoto
    if not result.empty:
        display(result)
        print(f'\nFor countries over threshold you have three options:')
        print('- Find values from official gov source')
        print('- Remove countries')
        print('- Include them to prediction (it will come less accurate than real val)')
    else:
        print("No country over threshold")
    
    return result

def perc20col_zeronan(df, threshold=0.3):
    colonne_numeriche = df.select_dtypes(include=[np.number]).columns
    risultati = {}
    print("Cerca i paesi per i quali più del 20% delle colonne numeriche hanno più del 30% di zeri o nan mancanti")
    na_or_zero = input('Inserire "==0" per cercare zeri, o ".isna()" per cercare valori mancanti: ')
    
    for country in df['country'].unique():
        df_country = df[df['country'] == country]
        colonne_con_zeri = []
        for col in colonne_numeriche:
            if na_or_zero == "==0":
                condizione = (df_country[col] == 0)
            elif na_or_zero == ".isna()":
                condizione = df_country[col].isna()
            else:
                print("Input non valido. Si prega di inserire '==0' o '.isna()'")
                return
            if condizione.sum() / len(df_country[col]) > threshold:
                colonne_con_zeri.append(col)
        if len(colonne_con_zeri) / len(colonne_numeriche) > 0.2:
            risultati[country] = colonne_con_zeri

    # Controlla se ci sono risultati e stampa l'output appropriato
    if risultati:
        print("\nPaesi per i quali più del 20% delle colonne numeriche hanno più del 30% di zeri o valori mancanti:")
        for country, colonne in risultati.items():
            print(f"{country}: {len(colonne)} colonne\n")
    else:
        print("Nessun paese trovato.\n")

    return risultati

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from cleaning import prepare_tbi_data
import subprocess
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


def plot_before_cleaning(df_original):


    # plot Top NA
    na_counts = df_original.isnull().sum()

    na_counts = na_counts[na_counts > 0]

    na_counts = na_counts.sort_values(ascending=False)

    top_20_na = na_counts.head(20)

    plt.figure(figsize=(14, 8)) 

    # Use Seaborn for enhanced aesthetics
    sns.barplot(x=top_20_na.index, y=top_20_na.values, palette="viridis")  # Choose a suitable color palette

    plt.title('Top 20 Columns with the Most NA Values', fontsize=18, fontweight='bold')  
    plt.xlabel('Column Names', fontsize=14)
    plt.ylabel('Number of NA Values', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=12) 
    plt.yticks(fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7) 
    sns.despine()
    plt.tight_layout()

    output_dir = '../figs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    output_path = os.path.join(output_dir, 'top_20_na_values.png') 

    plt.savefig(output_path, dpi=300)  



    # plot outliers
    data = df_original.iloc[:,1:]

    unique_values_data = {}
    for col in data.columns:
        unique_vals = data[col].unique()

        unique_values_data[col] = unique_vals

    data_list = []
    for col, unique_vals in unique_values_data.items():
        for val in unique_vals:
            data_list.append({'column': col, 'value': val})

    df = pd.DataFrame(data_list)

    df['column_code'] = df['column'].astype('category').cat.codes

    plt.figure(figsize=(16, 8))
    sns.scatterplot(x='column_code', y='value', data=df, s=50)

    plt.xlabel('Columns', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('Unique Values Across Columns - Check Outliers', fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'values_across_col.png') 

    plt.savefig(output_path, dpi=300) 

    return

def plot_after_cleaning(prepared_data, df_original):

    prepared_data['ActNorm_inverse'] = 1-prepared_data['ActNorm']
    prepared_data['GCSGroup_01'] = 2 - prepared_data['GCSGroup']

    # plot heatmap
    selected_variables = ['High_impact_InjSev','Amnesia_verb','Seiz', 'Vomit', 'HA_verb','Dizzy', 'ActNorm_inverse','GCSTotal', 'AMS', 'SFxBas', 'Hema', 'Clav', 'NeuroD', 'OSI', 'PosCT', 'PosIntFinal']
    selected_data = prepared_data[selected_variables]
    correlation_matrix = selected_data.corr()
    plt.figure(figsize=(10, 8))
    cmap = sns.color_palette("RdBu_r", 100)

    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=.5, linecolor="white", cmap=cmap,
                        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})


    plt.title("Correlation Heatmap", fontweight='bold') 
    plt.xticks(rotation=45, ha="right", fontsize=10) 
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel("Variables", fontsize=12)
    plt.ylabel("Variables", fontsize=12)

    plt.tight_layout()
    plt.savefig("../figs/eda3.png", dpi=300, bbox_inches='tight')


    citbi_data = prepared_data[prepared_data['PosIntFinal'] == 1]
    cttbi_data = prepared_data[prepared_data['PosCT'] == 1]
    notbi_data = prepared_data[(prepared_data['PosCT'] != 1) & (prepared_data['PosIntFinal'] != 1) ]

    columns_to_plot = ['Seiz', 'Vomit', 'HA_verb','Dizzy', 'ActNorm_inverse']

    columns = ['Amnesia_verb','Seiz', 'Vomit', 'HA_verb','Dizzy', 'ActNorm_inverse']

    # Plot pie
    frequencies1 = {}
    frequencies2 = {}
    for col in columns:
        frequencies1[col] = citbi_data[col].value_counts(normalize=True).get(1, 0) 
        frequencies2[col] = notbi_data[col].value_counts(normalize=True).get(1, 0)

    frequencies1_df = pd.DataFrame.from_dict(frequencies1, orient='index', columns=['Frequency'])
    frequencies2_df = pd.DataFrame.from_dict(frequencies2, orient='index', columns=['Frequency'])
    #frequencies_df = frequencies_df.sort_values(by='Frequency', ascending=False)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # 1 行 2 列
    label_fontsize = 17 
    autopct_fontsize = 15  

    ax1.pie(frequencies1_df['Frequency'],
            labels=frequencies1_df.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('pastel'),
            textprops={'fontsize': label_fontsize})  
    ax1.set_title("Proportion of symptoms in ciTBI", fontsize=17,fontweight='bold')
    ax1.axis('equal')


    ax2.pie(frequencies2_df['Frequency'],
            labels=frequencies2_df.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('pastel'),
            textprops={'fontsize': label_fontsize}) 
    ax2.set_title("Proportion of symptoms in noTBI", fontsize=17,fontweight='bold')
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig("../figs/eda2.png", dpi=300, bbox_inches='tight')


    # plot eda4
    orderedCT_tbi_data = prepared_data[(prepared_data['CTForm1'] == 1) & (prepared_data['PosIntFinal'] == 1)]
    orderedCT_data = prepared_data[prepared_data['CTForm1'] == 1]
    columns = ['IndAge', 'IndAmnesia', 'IndAMS', 'IndClinSFx', 'IndHA', 'IndHema', 'IndLOC','IndMech', 'IndNeuroD', 'IndRqstMD','IndRqstParent','IndRqstTrauma','IndSeiz','IndVomit','IndXraySFx','IndOth']

    frequencies1 = {}
    frequencies2 = {}
    for col in columns:
        frequencies1[col] = orderedCT_tbi_data[col].value_counts(normalize=True).get(1, 0)
        frequencies2[col] = orderedCT_data[col].value_counts(normalize=True).get(1, 0)

    frequencies1_df = pd.DataFrame.from_dict(frequencies1, orient='index', columns=['Frequency'])
    frequencies2_df = pd.DataFrame.from_dict(frequencies2, orient='index', columns=['Frequency'])

    frequencies1_df['Group'] = 'ciTBI'
    frequencies2_df['Group'] = 'all ordered'
    frequencies_df = pd.concat([frequencies1_df, frequencies2_df])
    frequencies_df = frequencies_df.reset_index().rename(columns={'index': 'Exam Result'})


    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid") 

    ax = sns.barplot(x='Exam Result', y='Frequency', hue='Group', data=frequencies_df, palette='pastel')

    plt.xlabel('Indication of ordering CT', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    plt.title("Proportion of the Indications of ordering CT in patients ordered CT", fontsize=17,fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12) 
    plt.ylim(0, frequencies_df['Frequency'].max() * 1.1) 
    plt.tight_layout() 
    plt.legend(title='Group')

    plt.savefig("../figs/eda4.png", dpi=300, bbox_inches='tight')

    # plot eda1

    tbi_data = prepared_data[prepared_data['PosIntFinal'] == 1]


    col_name = 'InjuryMech'
    sub_titiles = ['Fall from an elevation',' Fall to ground from standing/walking/running','Occupant in motor vehicle collision']
    nums = [8,6,1]

    sns.set(context='paper', style='white', font='sans-serif', font_scale=1.2)
    # plt.rcParams['font.family'] = 'Arial'  
    plt.rcParams['axes.linewidth'] = 0.8
    #plt.rcParams['grid.linestyle'] = '--'  
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4

    fig, axes = plt.subplots(1,3, figsize=(10, 3)) 
    axes = axes.flatten()


    colors = sns.color_palette("tab10", 3) 

    for i, num in enumerate(nums):

        age_injury_proportion = prepared_data.groupby('AgeinYears')[col_name].apply(lambda x: (x == num).sum() / len(x))

        sns.lineplot(x=age_injury_proportion.index, y=age_injury_proportion.values, marker='o', ax=axes[i], color='black', linewidth=1, markersize=3)
        axes[i].set_title(f'{sub_titiles[i]}', fontsize=10)
        axes[i].set_xlabel('Age (Years)', fontsize=10)
        axes[i].set_ylabel('Proportion', fontsize=10)
        # axes[i].grid(False, linewidth=0.5, linestyle='--')  
        axes[i].tick_params(axis='x', rotation=45, labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        


    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.suptitle('Proportion of Findings by Age among ciTBI', fontsize=14)
    plt.savefig("../figs/eda1.png", dpi=300, bbox_inches='tight')

    
    # plot finding2
    young_data = prepared_data[prepared_data['AgeinYears'] <= 2]
    older_data = prepared_data[prepared_data['AgeinYears'] > 2]


    columns = ['GCSGroup_01', 'AMS', 'SFxBas', 'Hema', 'Clav', 'NeuroD', 'OSI']
    frequencies1 = {}
    frequencies2 = {}
    for col in columns:
        frequencies1[col] = young_data[col].value_counts(normalize=True).get(1, 0)
        frequencies2[col] = older_data[col].value_counts(normalize=True).get(1, 0)

    frequencies1_df = pd.DataFrame.from_dict(frequencies1, orient='index', columns=['Frequency'])
    frequencies2_df = pd.DataFrame.from_dict(frequencies2, orient='index', columns=['Frequency'])

    frequencies1_df['Group'] = 'Less than 2 years old'
    frequencies2_df['Group'] = 'More than 2 years old'
    frequencies_df = pd.concat([frequencies1_df, frequencies2_df])
    frequencies_df = frequencies_df.reset_index().rename(columns={'index': 'Exam Result'})



    fig, axes = plt.subplots(2, 1, figsize=(12, 12))  

    sns.set(style="whitegrid")
    ax1 = sns.barplot(x='Exam Result', y='Frequency', hue='Group', data=frequencies_df, palette='pastel', ax=axes[0]) # 指定在 axes[0] 上绘制
    ax1.set_xlabel('', fontsize=14)
    ax1.set_ylabel('Proportion', fontsize=14)
    ax1.set_title("Proportion of exam results group by age in all observations", fontsize=17, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.set_ylim(0, frequencies_df['Frequency'].max() * 1.1)
    ax1.legend(title='Group', fontsize=13)


    young_data = prepared_data[(prepared_data['AgeinYears'] <= 2) & (prepared_data['PosIntFinal'] == 1)]
    older_data = prepared_data[(prepared_data['AgeinYears'] > 2) & (prepared_data['PosIntFinal'] == 1)]


    columns = ['GCSGroup_01', 'AMS', 'SFxBas', 'Hema', 'Clav', 'NeuroD', 'OSI']
    frequencies1 = {}
    frequencies2 = {}
    for col in columns:
        frequencies1[col] = young_data[col].value_counts(normalize=True).get(1, 0)
        frequencies2[col] = older_data[col].value_counts(normalize=True).get(1, 0)

    frequencies1_df = pd.DataFrame.from_dict(frequencies1, orient='index', columns=['Frequency'])
    frequencies2_df = pd.DataFrame.from_dict(frequencies2, orient='index', columns=['Frequency'])

    frequencies1_df['Group'] = 'Less than 2 years old'
    frequencies2_df['Group'] = 'More than 2 years old'
    frequencies_df = pd.concat([frequencies1_df, frequencies2_df])
    frequencies_df = frequencies_df.reset_index().rename(columns={'index': 'Exam Result'})

    ax2 = sns.barplot(x='Exam Result', y='Frequency', hue='Group', data=frequencies_df, palette='pastel', ax=axes[1]) 
    ax2.set_xlabel('Positive Exam Result', fontsize=14)
    ax2.set_ylabel('Propotion', fontsize=14)
    ax2.set_title("Proportion of Positive Exam Results group by age in ciTBI", fontsize=17, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=12)
    ax2.set_ylim(0, frequencies_df['Frequency'].max() * 1.1)
    ax2.legend(title='Group', fontsize=13)

    plt.tight_layout()  

    plt.savefig("../figs/finding2.png", dpi=300, bbox_inches='tight')

    # plot finding1
    symptoms_columns = ['Amnesia_verb','Seiz', 'Vomit', 'HA_verb','Dizzy', 'ActNorm_inverse']
    prepared_data['num_symptoms_comb'] = prepared_data[symptoms_columns].sum(axis=1)
    citbi_data = prepared_data[prepared_data['PosIntFinal'] == 1]
    notbi_data = prepared_data[(prepared_data['PosIntFinal'] != 1)&(prepared_data['PosCT'] != 1)]

    plt.figure(figsize=(10, 6))  
    sns.set(style="whitegrid") 

    ax = plt.gca() 
    ax.grid(axis='y', alpha=0.5) 
    ax.xaxis.grid(False) 
    sns.histplot(citbi_data['num_symptoms_comb'],  label="ciTBI patients", kde=False, stat="probability", color=sns.color_palette('pastel')[0]) 
    sns.histplot(notbi_data['num_symptoms_comb'],  label="noTBI patients", kde=False, stat="probability", color=sns.color_palette('pastel')[1])    
    plt.xlabel("Number of Symptoms")  
    plt.ylabel("Proportion") 
    plt.title("Distribution of Patients' Symptom Count",fontweight='bold')  
    plt.legend()  
    # plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("../figs/finding1.png", dpi=300, bbox_inches='tight')

    # plot finding3
    tbi_data = prepared_data[prepared_data['PosIntFinal'] == 1]
    col_names = ['Finding2', 'Finding3', 'Finding5', 'Finding7']
    sub_titiles = ['cerebral contusion','cerebral edema','diastasis of the skull', 'extra-axial hematoma']

    sns.set(context='paper', style='white', font='sans-serif', font_scale=1.2)
    # plt.rcParams['font.family'] = 'Arial'  
    plt.rcParams['axes.linewidth'] = 0.8
    #plt.rcParams['grid.linestyle'] = '--'  
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4

    fig, axes = plt.subplots(2, 2, figsize=(10, 7)) 
    axes = axes.flatten()


    colors = sns.color_palette("tab10", len(col_names)) 

    for i, col_name in enumerate(col_names):
        
        age_injury_proportion = tbi_data.groupby('AgeinYears')[col_name].apply(lambda x: (x == 1).sum() / len(x))

        sns.lineplot(x=age_injury_proportion.index, y=age_injury_proportion.values, marker='o', ax=axes[i], color='black', linewidth=1, markersize=3)
        axes[i].set_title(f'{sub_titiles[i]}', fontsize=12)
        axes[i].set_xlabel('Age (Years)', fontsize=10)
        axes[i].set_ylabel('Proportion', fontsize=10)
        # axes[i].grid(False, linewidth=0.5, linestyle='--') 
        axes[i].tick_params(axis='x', rotation=45, labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].spines['top'].set_visible(True)
        axes[i].spines['right'].set_visible(True)
        

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Proportion of CT Findings by Age among ciTBI', fontsize=14, fontweight='bold')
    plt.savefig("../figs/finding3.png", dpi=300, bbox_inches='tight')

    # stab check
    prepared_data_p = prepare_tbi_data(df_original, check_consistency=False)
    young_data = prepared_data[prepared_data['AgeinYears'] <= 2]
    older_data = prepared_data[prepared_data['AgeinYears'] > 2]


    columns = ['GCSGroup_01', 'AMS', 'SFxBas', 'Hema', 'Clav', 'NeuroD', 'OSI']
    frequencies1 = {}
    frequencies2 = {}
    for col in columns:
        frequencies1[col] = young_data[col].value_counts(normalize=True).get(1, 0)
        frequencies2[col] = older_data[col].value_counts(normalize=True).get(1, 0)

    frequencies1_df = pd.DataFrame.from_dict(frequencies1, orient='index', columns=['Frequency'])
    frequencies2_df = pd.DataFrame.from_dict(frequencies2, orient='index', columns=['Frequency'])


    frequencies1_df['Group'] = 'Less than 2 years old'
    frequencies2_df['Group'] = 'More than 2 years old'
    frequencies_df = pd.concat([frequencies1_df, frequencies2_df])
    frequencies_df = frequencies_df.reset_index().rename(columns={'index': 'Exam Result'})

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))  

    sns.set(style="whitegrid")
    ax1 = sns.barplot(x='Exam Result', y='Frequency', hue='Group', data=frequencies_df, palette='pastel', ax=axes[0]) # 指定在 axes[0] 上绘制
    ax1.set_xlabel('', fontsize=14)
    ax1.set_ylabel('Proportion', fontsize=14)
    ax1.set_title("Proportion of exam results group by age in all observations", fontsize=17, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.set_ylim(0, frequencies_df['Frequency'].max() * 1.1)
    ax1.legend(title='Group', fontsize=13)


    young_data = prepared_data[(prepared_data['AgeinYears'] <= 2) & (prepared_data['PosIntFinal'] == 1)]
    older_data = prepared_data[(prepared_data['AgeinYears'] > 2) & (prepared_data['PosIntFinal'] == 1)]


    columns = ['GCSGroup_01', 'AMS', 'SFxBas', 'Hema', 'Clav', 'NeuroD', 'OSI']
    frequencies1 = {}
    frequencies2 = {}
    for col in columns:
        frequencies1[col] = young_data[col].value_counts(normalize=True).get(1, 0)
        frequencies2[col] = older_data[col].value_counts(normalize=True).get(1, 0)

    frequencies1_df = pd.DataFrame.from_dict(frequencies1, orient='index', columns=['Frequency'])
    frequencies2_df = pd.DataFrame.from_dict(frequencies2, orient='index', columns=['Frequency'])


    frequencies1_df['Group'] = 'Less than 2 years old'
    frequencies2_df['Group'] = 'More than 2 years old'
    frequencies_df = pd.concat([frequencies1_df, frequencies2_df])
    frequencies_df = frequencies_df.reset_index().rename(columns={'index': 'Exam Result'})

    ax2 = sns.barplot(x='Exam Result', y='Frequency', hue='Group', data=frequencies_df, palette='pastel', ax=axes[1]) # 指定在 axes[1] 上绘制
    ax2.set_xlabel('Positive Exam Result', fontsize=14)
    ax2.set_ylabel('Propotion', fontsize=14)
    ax2.set_title("Proportion of Positive Exam Results group by age in ciTBI", fontsize=17, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=12)
    ax2.set_ylim(0, frequencies_df['Frequency'].max() * 1.1)
    ax2.legend(title='Group', fontsize=13)

    plt.tight_layout() 
    plt.savefig("../figs/finding2-p.png", dpi=300, bbox_inches='tight')


    # model plot
    drop_col = ['PatNum', 'EmplType', 'Certification', 'CTDone', 'PosCT', 'Race', 'Ethnicity', 'HospHeadPosCT', 'Intub24Head', 'Neurosurgery',
            'HospHead', 'DeathTBI','EDCT','CTSed'] + [f"Finding{i}" for i in range(1, 15)] + [f"Finding{i}" for i in range(20, 24)] 
    data1 = prepared_data.drop(drop_col, axis=1)
    data1= data1.dropna()

    X1 = data1.drop('PosIntFinal', axis=1)
    y1 = data1['PosIntFinal']
    X1_scaled = StandardScaler().fit_transform(X1)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)

    data2 = prepared_data_p.drop(drop_col, axis=1)
    data2= data2.dropna()
    X2 = data2.drop('PosIntFinal', axis=1)
    y2 = data2['PosIntFinal']
    X2_scaled = StandardScaler().fit_transform(X2)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42)


    # --- Model Training (Data1) ---
    model1 = LogisticRegression(random_state=42, penalty='l2', solver='liblinear', class_weight='balanced', C=0.1)
    model1.fit(X1_train, y1_train)
    y1_pred_proba = model1.predict_proba(X1_test)[:, 1]

    # --- Model Training (Data2) ---
    model2 = LogisticRegression(random_state=42, penalty='l2', solver='liblinear', class_weight='balanced', C=0.1)
    model2.fit(X2_train, y2_train)
    y2_pred_proba = model2.predict_proba(X2_test)[:, 1]

    # --- ROC Curve Calculation ---
    fpr1, tpr1, _ = roc_curve(y1_test, y1_pred_proba)
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(y2_test, y2_pred_proba)
    roc_auc2 = auc(fpr2, tpr2)

    # --- Plotting with Seaborn ---
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="darkgrid")
    sns.set(context='paper', style='white', font='sans-serif', font_scale=1.2)

    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'Original ROC curve (area = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, color='green', lw=2, label=f'Perturbed ROC curve (area = {roc_auc2:.2f})')  # Different color for Data2
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.tight_layout()  
    plt.savefig("../figs/model-p.png", dpi=300, bbox_inches='tight')

    return



def tex_to_pdf(tex_file, output_dir="."):
    """
    convert text to pdf
    """

    try:
       
        os.makedirs(output_dir, exist_ok=True)

        command = ["pdflatex",
                   "-output-directory", output_dir,
                   tex_file]

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print("LaTeX compilation successful.")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("LaTeX compilation failed:")
        print(e.stderr)
        raise  

    except FileNotFoundError:
        print("Error: pdflatex not found.  Make sure LaTeX is installed and in your PATH.")
        raise
import click
import pandas as pd
import numpy as np
from scipy.stats import skewnorm, norm

import uuid
import json
import copy

FEATURES = ["rupture", "sex", "age", "hta", "bmi", "sporadic_case", "multiple_IA", "IA_location", "adjusted_size_ratio", 
            "tobacco", "alcohol", "headaches", "diabetes", "dyslipidemia", "ischemic_stroke_history", 
            "ischemic_heart_disease_history", "pad_history", "carotid_artery_stenosis_history", "aortic_aneurysm_history", 
            "statin_ttt", "platelet_aggregation_inhibiting_ttt", "vka_or_anticoagulant_ttt", "anti_inflammatory_ttt", 
            "hormone_therapy_ttt", "allergy", "asthma", "atopy", "eczema"] 

def yes_no_var(percentage_yes, nb_samples) :
    return np.random.choice([1, 0], nb_samples, p=[percentage_yes, 1-percentage_yes])

def generate_dataframe(nb_sample):
    np.random.seed(111)
    df = pd.DataFrame()
    nb_samples = nb_sample
    nb_choice = nb_samples * 5

    for f in FEATURES :
        if "rupture" in f:
            df[f] = yes_no_var(0.397, nb_samples)
        
        if 'sex' in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = np.random.choice([1, 0], len(df[df.rupture == 1]), p=[0.30, 0.70])
            var[df[df.rupture == 0].index] = np.random.choice([1, 0], len(df[df.rupture == 0]), p=[0.27, 0.73])
            df[f] = var
            
        if 'age' in f:
            var = np.repeat(None, nb_samples)
            mean = 50.3 ; sd = 12.1 ; min_val = 18 ; max_val = 87
            random_nb = np.random.normal(loc = mean, scale = sd, size = nb_choice).astype(int)
            random_nb = random_nb[np.where((random_nb >= min_val)&(random_nb <= max_val))]
            var[df[df.rupture == 1].index] = np.random.choice(random_nb, size = len(df[df.rupture == 1]))
            mean = 55.8 ; sd = 11.9 ; min_val = 20 ; max_val = 85
            random_nb = np.random.normal(loc = mean, scale = sd, size = nb_choice).astype(int)
            random_nb_No = random_nb[np.where((random_nb >= min_val)&(random_nb <= max_val))]
            var[df[df.rupture == 0].index] = np.random.choice(random_nb, size = len(df[df.rupture == 0]))
            df[f] = var
            
        if 'sporadic_case' in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = yes_no_var(0.84, len(df[df.rupture == 1]))
            var[df[df.rupture == 0].index] = yes_no_var(0.81, len(df[df.rupture == 0]))
            df[f] = var
            
        if 'multiple_IA' in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = yes_no_var(0.34, len(df[df.rupture == 1]))
            var[df[df.rupture == 0].index] = yes_no_var(0.30, len(df[df.rupture == 0]))
            df[f] = var
            
        if "IA_location" in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = np.random.choice([0, 1, 2, 3], len(df[df.rupture == 1]), p=[0.12, 0.23, 0.39, 0.26])
            var[df[df.rupture == 0].index] = np.random.choice([0, 1, 2, 3], len(df[df.rupture == 0]), p=[0.33, 0.29, 0.24, 0.14])
            df[f] = var
            
        if 'adjusted_size_ratio' in f:
            select_indiv = []
            a = 2.27 ; loc = 0.97 ; scale = 1.29
            random_nb = skewnorm.rvs(a, loc, scale, size = nb_choice).round(decimals = 2)
            random_nb = random_nb[np.where(random_nb >=0.25)]
            size_ICA = np.random.choice(random_nb, size = len(df[df["IA_location"] == 0]))

            select_indiv = []
            a = 2.88 ; loc = 1.17 ; scale = 1.44
            random_nb = skewnorm.rvs(a, loc, scale, size = nb_choice).round(decimals = 2)
            random_nb = random_nb[np.where(random_nb >=0.33)]
            size_MCA = np.random.choice(random_nb, size = len(df[df["IA_location"] == 1]))

            select_indiv = []
            a = 7.52 ; loc = 3.05 ; scale = 4.04
            random_nb = skewnorm.rvs(a, loc, scale, size = nb_choice).round(decimals = 2)
            random_nb = random_nb[np.where(random_nb >=1)]
            size_ACA = np.random.choice(random_nb, size = len(df[df["IA_location"] == 2]))

            select_indiv = []
            a = 2.34 ; loc = 2.48 ; scale = 3.56
            random_nb = skewnorm.rvs(a, loc, scale, size = nb_choice).round(decimals = 2)
            random_nb = random_nb[np.where(random_nb >=0.66)]
            size_PCA = np.random.choice(random_nb, size = len(df[df["IA_location"] == 3]))

            size = np.repeat(-1.11, nb_samples)
            size[df[df["IA_location"] == 0].index] = size_ICA
            size[df[df["IA_location"] == 1].index] = size_MCA
            size[df[df["IA_location"] == 2].index] = size_ACA
            size[df[df["IA_location"] == 3].index] = size_PCA
            df[f] = size
            
        if "tobacco" in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = np.random.choice([0, 1, 2], len(df[df.rupture == 1]), p=[0.29, 0.46, 0.25])
            var[df[df.rupture == 0].index] = np.random.choice([0, 1, 2], len(df[df.rupture == 0]), p=[0.29, 0.42, 0.29])
            df[f] = var
        
        if "alcohol" in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = np.random.choice([0, 1], len(df[df.rupture == 1]), p=[0.87, 0.13])
            var[df[df.rupture == 0].index] = np.random.choice([0, 1], len(df[df.rupture == 0]), p=[0.89, 0.11])
            df[f] = var
            
        if "hta" in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = yes_no_var(0.36, len(df[df.rupture == 1]))
            var[df[df.rupture == 0].index] = yes_no_var(0.38, len(df[df.rupture == 0]))
            df[f] = var
            
        if "headaches" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.hta == 1)].index] = yes_no_var(0.15, len(df[(df.rupture == 1)&(df.hta == 1)]))
            var[df[(df.rupture == 1)&(df.hta == 0)].index] = yes_no_var(0.15, len(df[(df.rupture == 1)&(df.hta == 0)]))
            var[df[(df.rupture == 0)&(df.hta == 1)].index] = yes_no_var(0.13, len(df[(df.rupture == 0)&(df.hta == 1)]))
            var[df[(df.rupture == 0)&(df.hta == 0)].index] = yes_no_var(0.22, len(df[(df.rupture == 0)&(df.hta == 0)]))
            df[f] = var
        
        if 'bmi' in f:
            means = [25.4, 0] # bmi N(25.4, 23.04) , diabetes N(0,1), cov(bmi, diabetes) = 3
            covs = [[23.04, 3],
                    [3, 1]]
            
            # Generate data
            data = np.random.multivariate_normal(means, covs, nb_choice)
            random_bmi = data[:,0]
            random_diabetes = data[:,1]

            # Adjust mix and max values for bmi
            min_val = 15.6 ; max_val = 54.8
            mask = np.where((random_bmi >= min_val)&(random_bmi <= max_val))
            random_bmi = random_bmi[mask]
            random_diabetes = random_diabetes[mask]

            # Select the desired number of samples
            idxs = np.random.choice(np.arange(len(random_bmi)), nb_samples, replace=False)
            random_bmi = random_bmi[idxs]
            random_diabetes = random_diabetes[idxs]

            # Discretize the discrete diabetes variable
            mask1 = (df.hta == 0)&(random_diabetes >= norm.ppf(1 - 0.02))
            mask2 = (df.rupture == 1)&(df.hta == 1)&(random_diabetes >= norm.ppf(1 - 0.06))
            mask3 = (df.rupture == 0)&(df.hta == 1)&(random_diabetes >= norm.ppf(1 - 0.1))
            mask = mask1 | mask2 | mask3
            random_diabetes[mask] = 1
            random_diabetes[~mask] = 0

            df['bmi'] = random_bmi
            df['diabetes'] = random_diabetes
            
        if "dyslipidemia" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.hta == 1)].index] = yes_no_var(0.30, len(df[(df.rupture == 1)&(df.hta == 1)]))
            var[df[(df.rupture == 1)&(df.hta == 0)].index] = yes_no_var(0.12, len(df[(df.rupture == 1)&(df.hta == 0)]))
            var[df[(df.rupture == 0)&(df.hta == 1)].index] = yes_no_var(0.40, len(df[(df.rupture == 0)&(df.hta == 1)]))
            var[df[(df.rupture == 0)&(df.hta == 0)].index] = yes_no_var(0.16, len(df[(df.rupture == 0)&(df.hta == 0)]))
            df[f] = var
        
        if "ischemic_stroke_history" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.dyslipidemia == 1)].index] = yes_no_var(0.08, len(df[(df.rupture == 1)&(df.dyslipidemia == 1)]))
            var[df[(df.rupture == 1)&(df.dyslipidemia == 0)].index] = yes_no_var(0.02, len(df[(df.rupture == 1)&(df.dyslipidemia == 0)]))
            var[df[(df.rupture == 0)&(df.dyslipidemia == 1)].index] = yes_no_var(0.20, len(df[(df.rupture == 0)&(df.dyslipidemia == 1)]))
            var[df[(df.rupture == 0)&(df.dyslipidemia == 0)].index] = yes_no_var(0.07, len(df[(df.rupture == 0)&(df.dyslipidemia == 0)]))
            df[f] = var
            
        if "ischemic_heart_disease_history" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.ischemic_stroke_history == 1)].index] = yes_no_var(0.09, len(df[(df.rupture == 1)&(df.ischemic_stroke_history == 1)]))
            var[df[(df.rupture == 1)&(df.ischemic_stroke_history == 0)].index] = yes_no_var(0.01, len(df[(df.rupture == 1)&(df.ischemic_stroke_history == 0)]))
            var[df[(df.rupture == 0)&(df.ischemic_stroke_history == 1)].index] = yes_no_var(0.12, len(df[(df.rupture == 0)&(df.ischemic_stroke_history == 1)]))
            var[df[(df.rupture == 0)&(df.ischemic_stroke_history == 0)].index] = yes_no_var(0.04, len(df[(df.rupture == 0)&(df.ischemic_stroke_history == 0)]))
            df[f] = var
            
        if "pad_history" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 1)].index] = yes_no_var(0.33, len(df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 1)]))
            var[df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 0)].index] = yes_no_var(0.01, len(df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 0)]))
            var[df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 1)].index] = yes_no_var(0.16, len(df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 1)]))
            var[df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 0)].index] = yes_no_var(0.02, len(df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 0)]))
            df[f] = var
            
        if "carotid_artery_stenosis_history" in f:
            df[f] = yes_no_var(0.03, nb_samples)
            
        if 'aortic_aneurysm_history' in f:
            df[f] = yes_no_var(0.005, nb_samples)
            
        if "statin_ttt" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.dyslipidemia == 1)].index] = yes_no_var(0.46, len(df[(df.rupture == 1)&(df.dyslipidemia == 1)]))
            var[df[(df.rupture == 1)&(df.dyslipidemia == 0)].index] = yes_no_var(0.01, len(df[(df.rupture == 1)&(df.dyslipidemia == 0)]))
            var[df[(df.rupture == 0)&(df.dyslipidemia == 1)].index] = yes_no_var(0.39, len(df[(df.rupture == 0)&(df.dyslipidemia == 1)]))
            var[df[(df.rupture == 0)&(df.dyslipidemia == 0)].index] = yes_no_var(0.03, len(df[(df.rupture == 0)&(df.dyslipidemia == 0)]))
            df[f] = var
            
        if "platelet_aggregation_inhibiting_ttt" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 1)].index] = yes_no_var(0.5, len(df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 1)]))
            var[df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 0)].index] = yes_no_var(0.09, len(df[(df.rupture == 1)&(df.ischemic_heart_disease_history == 0)]))
            var[df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 1)].index] = yes_no_var(0.24, len(df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 1)]))
            var[df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 0)].index] = yes_no_var(0.13, len(df[(df.rupture == 0)&(df.ischemic_heart_disease_history == 0)]))
            df[f] = var
            
        if "vka_or_anticoagulant_ttt" in f:
            df[f] = yes_no_var(0.03, nb_samples)
            
        if "anti_inflammatory_ttt" in f:
            df[f] = yes_no_var(0.03, nb_samples)
            
        if 'hormone_therapy_ttt' in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = yes_no_var(0.02, len(df[df.rupture == 1]))
            var[df[df.rupture == 0].index] = yes_no_var(0.04, len(df[df.rupture == 0]))
            df[f] = var
        
        if "allergy" in f:
            var = np.repeat(None, nb_samples)
            var[df[df.rupture == 1].index] = yes_no_var(0.20, len(df[df.rupture == 1]))
            var[df[df.rupture == 0].index] = yes_no_var(0.25, len(df[df.rupture == 0]))
            df[f] = var
            
        if "asthma" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.allergy == 1)].index] = yes_no_var(0.16, len(df[(df.rupture == 1)&(df.allergy == 1)]))
            var[df[(df.rupture == 1)&(df.allergy == 0)].index] = yes_no_var(0.03, len(df[(df.rupture == 1)&(df.allergy == 0)]))
            var[df[(df.rupture == 0)&(df.allergy == 1)].index] = yes_no_var(0.18, len(df[(df.rupture == 0)&(df.allergy == 1)]))
            var[df[(df.rupture == 0)&(df.allergy == 0)].index] = yes_no_var(0.04, len(df[(df.rupture == 0)&(df.allergy == 0)]))
            df[f] = var
            
        if "atopy" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.allergy == 1)].index] = yes_no_var(0.15, len(df[(df.rupture == 1)&(df.allergy == 1)]))
            var[df[(df.rupture == 1)&(df.allergy == 0)].index] = yes_no_var(0.02, len(df[(df.rupture == 1)&(df.allergy == 0)]))
            var[df[(df.rupture == 0)&(df.allergy == 1)].index] = yes_no_var(0.14, len(df[(df.rupture == 0)&(df.allergy == 1)]))
            var[df[(df.rupture == 0)&(df.allergy == 0)].index] = yes_no_var(0.03, len(df[(df.rupture == 0)&(df.allergy == 0)]))
            df[f] = var
            
        if "eczema" in f:
            var = np.repeat(None, nb_samples)
            var[df[(df.rupture == 1)&(df.allergy == 1)].index] = yes_no_var(0.13, len(df[(df.rupture == 1)&(df.allergy == 1)]))
            var[df[(df.rupture == 1)&(df.allergy == 0)].index] = yes_no_var(0.04, len(df[(df.rupture == 1)&(df.allergy == 0)]))
            var[df[(df.rupture == 0)&(df.allergy == 1)].index] = yes_no_var(0.13, len(df[(df.rupture == 0)&(df.allergy == 1)]))
            var[df[(df.rupture == 0)&(df.allergy == 0)].index] = yes_no_var(0.04, len(df[(df.rupture == 0)&(df.allergy == 0)]))
            df[f] = var
    return df
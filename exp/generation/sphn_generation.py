import pandas as pd
import numpy as np
import joblib
import re
import uuid
from rdflib import ConjunctiveGraph
from string import Template
from scipy.stats import norm
from datetime import datetime, timedelta


def gen_sphn_kg(num_patients, timeOpt):
    df = pd.read_csv(f'data/syn_data_10000.csv', index_col=0)
    df.rename(columns={'output': 'outcome'}, inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    numerical = ['hospital_stay_length', 'gcs', 'nb_acte', 'age']
    categorical = ['gender', 'entry', 'entry_code', 'ica', 'ttt', 'ica_therapy', 'fever', 'o2_clinic', 'o2', 'hta', 'hct', 'tabagisme', 'etOH', 'diabete', 'headache', 'instable', 'vasospasme', 'ivh', 'outcome']

    events = ['nimodipine',  'paracetamol', 'nad', 'corotrop', 'morphine', 'dve', 'atl', 'iot']

    drug_events = [
        "nimodipine",
        "paracetamol",
        "nad",
        "corotrop",
        "morphine",
    ]

    proc_events = [
        "dve",
        "atl",
        "iot"
    ]

    events_codes = {
        "nimodipine": "C08CA06",  # ACT / drug administration event
        "paracetamol": "N02BE01",  # ACT / drug administration event
        "nad": "C01CA03",  # ACT / drug administration event
        "corotrop": "C01CE02",  # ACT / drug administration event
        "morphine": "N02AA01",  # ACT / drug administration event
        "dve": "00P6X0Z",  # Removal of Drainage Device from Cerebral Ventricle External Approach (ICD-10) / procedure
        "atl": "Z98.6",  # ICD-10 Drainage Device from Cerebral Ventricle External Approach (ICD-10) / procedure
        "iot": "0BH17EZ",  # ICD-10 / procedure thacheotomie
    }

    
    prefix = """   
    @prefix sphn: <http://sphn.org/> .
    @prefix nvasc: <http://nvasc.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    """

    sphn_age_template = Template(
        """
        nvasc:age_$age_id a sphn:Age ;
            sphn:hasQuantity [ rdf:type sphn:Quantity ;
                                sphn:hasValue "$age_value" ;
                                sphn:hasUnit "years" ] .
                                
        nvasc:synth_patient_$patient_id nvasc:hasAge nvasc:age_$age_id .
        """
    )

    if timeOpt == 'NT':
       
        sphn_procedure_template = Template(
        """
        nvasc:$proc_id a sphn:Pocedure ;
            rdfs:label "$proc_label"^^xsd:string ;
            sphn:hasCode nvasc:code_$proc_code .
            
        nvasc:synth_patient_$patient_id nvasc:hasProcedure nvasc:$proc_id .
        """
        )

        sphn_drug_administration_template = Template(
            """
            nvasc:$drug_adm_id a sphn:DrugAdministrationEvent ;
                rdfs:label "$drug_adm_label"^^xsd:string ;
                sphn:hasDrug nvasc:drug_$drug_code .
                
            nvasc:synth_patient_$patient_id nvasc:hasDrugAdministrationEvent nvasc:$drug_adm_id .
            """
        )
    else:
        sphn_procedure_template = Template(
        """
        nvasc:$proc_id a sphn:Pocedure ;
            rdfs:label "$proc_label"^^xsd:string ;
            sphn:hasCode nvasc:code_$proc_code ;
            sphn:hasStartDateTime "$proc_start_date"^^xsd:dateTime .
            
        nvasc:synth_patient_$patient_id nvasc:hasProcedure nvasc:$proc_id .
        """
        )

        sphn_drug_administration_template = Template(
            """
            nvasc:$drug_adm_id a sphn:DrugAdministrationEvent ;
                rdfs:label "$drug_adm_label"^^xsd:string ;
                sphn:hasDrug nvasc:drug_$drug_code ;
                sphn:hasStartDateTime "$drug_start_date"^^xsd:dateTime .
            
            nvasc:synth_patient_$patient_id nvasc:hasDrugAdministrationEvent nvasc:$drug_adm_id .
            """
        )

    sphn_gender_template = Template(
        """
        nvasc:gender_$gender_id a sphn:AdministrativeGender ;
            sphn:hasCode nvasc:code_$gender_code .
            
        nvasc:synth_patient_$patient_id nvasc:hasGender nvasc:gender_$gender_id .
        """
    )

    sphn_timed_diagnosis_code_template = Template(
        """
        nvasc:$diag_id a sphn:Diagnosis ;
            rdfs:label "$diag_label"^^xsd:string ;
            sphn:hasCode nvasc:code_$diag_code ;
            sphn:hasRecordDateTime "$diag_date"^^xsd:dateTime .
            
        nvasc:synth_patient_$patient_id nvasc:hasDiagnosis nvasc:$diag_id .
        """
    )

    sphn_diagnosis_code_template = Template(
        """
        nvasc:$diag_id a sphn:Diagnosis ;
            rdfs:label "$diag_label"^^xsd:string ;
            sphn:hasCode nvasc:code_$diag_code .
            
        nvasc:synth_patient_$patient_id nvasc:hasDiagnosis nvasc:$diag_id .
        """
    )

    sphn_diagnosis_quantity_template = Template(
        """
        nvasc:$diag_id a sphn:Diagnosis ;
            rdfs:label "$diag_label" ;
            sphn:hasQuantity [ rdf:type sphn:Quantity ;
                                sphn:hasValue "$diag_value" ;
                                sphn:hasUnit "$diag_unit" ] .
        
        nvasc:synth_patient_$patient_id nvasc:hasDiagnosis nvasc:$diag_id .
        """
    )

    nvasc_outcome = Template(
        """
        nvasc:synth_patient_$patient_id nvasc:hasOutcome nvasc:outcome_$outcome .
        """
    )


    def gen_start_event(y_min=2020, y_max=2023):
        n_days = (y_max - y_min) * 365
        d0 = datetime.fromisoformat(f"{y_min}-01-01")
        day_rand = round(np.random.uniform(n_days))
        delta = timedelta(
            days=day_rand,
            hours=round(norm.rvs(12, 5)),
            minutes=round(np.random.uniform(60)),
        )
        d_out = d0 + delta
        return d_out


    def gen_patient_rdf(row, kg):
        _i = row.name
        d_start = gen_start_event()
        for f in row.index:
            if f in drug_events:
                if row[f] != -1:
                    h = row[f]
                    if h == 0:
                        h = 1
                    d_event = d_start + timedelta(hours=h)
                    rdf = sphn_drug_administration_template.substitute(
                        drug_adm_id=uuid.uuid4(),
                        drug_adm_label=f,
                        drug_code=events_codes[f],
                        drug_start_date=d_event.isoformat(),
                        patient_id=_i,
                    )
                    kg.parse(data=prefix + rdf, format="turtle")
            elif f in proc_events:
                if row[f] != -1:
                    h = row[f]
                    if h == 0:
                        h = 1
                    d_event = d_start + timedelta(hours=h)
                    rdf = sphn_procedure_template.substitute(
                        proc_id=uuid.uuid4(),
                        proc_label=f,
                        proc_code=events_codes[f],
                        proc_start_date=d_event.isoformat(),
                        patient_id=_i,
                    )
                    kg.parse(data=prefix + rdf, format="turtle")

            elif f in numerical:
                value = None
                unit = None
                if f in ["age"]:
                    gender_value = row[f]
                    rdf = sphn_age_template.substitute(
                        patient_id=_i, age_id=_i, age_value=round(row[f]), age_determination_date=d_start.isoformat()
                    )
                    kg.parse(data=prefix + rdf, format="turtle")
                else:
                    if f == "hospital_stay_length":
                        value = round(row[f])
                        unit = "days"
                    elif f == "gcs":
                        value = round(row[f], 2)
                        unit = "gcs"
                    elif f == "nb_acte":
                        value = round(row[f])
                        unit = "received medical treatments"

                    rdf = sphn_diagnosis_quantity_template.substitute(
                        diag_id=uuid.uuid4(),
                        diag_label=f,
                        diag_value=value,
                        diag_unit=unit,
                        patient_id=_i,
                    )
                    kg.parse(data=prefix + rdf, format="turtle")

            elif f in categorical:
                if f in ["gender"]:
                    gender_value = row[f]
                    rdf = sphn_gender_template.substitute(
                        patient_id=_i, gender_id=_i, gender_code=gender_value
                    )
                    kg.parse(data=prefix + rdf, format="turtle")
                elif f in ["outcome"]:
                    outcome_value = row[f]
                    rdf = nvasc_outcome.substitute(outcome=outcome_value, patient_id=_i)
                    kg.parse(data=prefix + rdf, format="turtle")
                else:
                    diag_label = f
                    diag_code = row[f]
                    rdf = sphn_diagnosis_code_template.substitute(
                        diag_id=uuid.uuid4(),
                        diag_label=f,
                        diag_code=str(f) + "_" + str(row[f]),
                        patient_id=_i,
                    )
                    kg.parse(data=prefix + rdf, format="turtle")

    outcome_df = df.iloc[0:num_patients]
    no_outcome = outcome_df.drop(columns=["outcome"])

    ## Serialize data
    kg = ConjunctiveGraph()
    no_outcome.apply(gen_patient_rdf, axis=1, kg=kg)
    print(f"Generated {len(kg)} RDF triples")

    if timeOpt == 'NT':
        pass
    else:
        procedure_before_query = """
        CONSTRUCT {
            ?e1 time:before ?e2 .
        } WHERE {
            ?e1 sphn:hasStartDateTime ?start1 .
            ?e2 sphn:hasStartDateTime ?start2 .
            ?p nvasc:hasProcedure ?e1, ?e2 .
            
            filter((?start1 < ?start2) && (?e1 != ?e2))
        }
        """
        drug_before_query = """
        CONSTRUCT {
            ?e1 time:before ?e2 .
        } WHERE {
            ?e1 sphn:hasStartDateTime ?start1 .
            ?e2 sphn:hasStartDateTime ?start2 .
            ?p nvasc:hasDrugAdministrationEvent ?e1, ?e2 .
            
            filter((?start1 < ?start2) && (?e1 != ?e2))
        }
        """
        diag_before_query = """
        CONSTRUCT {
            ?e1 time:before ?e2 .
        } WHERE {
            ?e1 sphn:hasRecordDateTime ?start1 .
            ?e2 sphn:hasRecordDateTime ?start2 .
            ?p nvasc:hasDiagnosis ?e1, ?e2 .
            
            filter((?start1 < ?start2) && (?e1 != ?e2))
        }
        """
        res = kg.query(procedure_before_query)
        for t in res:
            kg.add(t)
        res = kg.query(drug_before_query)
        for t in res:
            kg.add(t)
        print(f"Generated {len(kg)} RDF triples")

    if timeOpt == 'NT':
        print(f"KG length = {len(kg)} RDF triples")
        kg.serialize(f"data/sphn_pc_NT_{num_patients}.nt", format="ntriples")
        joblib.dump(df["outcome"].astype(int).to_list(), f"data/outcomes_pc_NT_{num_patients}.joblib")
    elif timeOpt == 'TR':
        # Delete timestamps.
        delete_ts_query = """
        DELETE {
            ?context1 sphn:hasDeterminationDateTime ?determin .
            ?context2 sphn:hasStartDateTime ?start .
        } WHERE {
            ?context1 sphn:hasDeterminationDateTime ?determin .
            ?context2 sphn:hasStartDateTime ?start .
        }
        """
        kg.update(delete_ts_query)
        print(f"KG length = {len(kg)} RDF triples")
        kg.serialize(f"data/sphn_pc_TR_{num_patients}.nt", format="ntriples")
        joblib.dump(df["outcome"].astype(int).to_list(), f"data/outcomes_pc_TR_{num_patients}.joblib")
    elif timeOpt == 'TS':
        # Delete time relations.
        delete_tr_query = """
        DELETE {
            ?e1 time:before ?e2 .
        } WHERE {
            ?e1 time:before ?e2 .
        }
        """
        kg.update(delete_tr_query)
        print(f"KG length = {len(kg)} RDF triples")
        kg.serialize(f"data/sphn_pc_TS_{num_patients}.nt", format="ntriples")
        joblib.dump(df["outcome"].astype(int).to_list(), f"data/outcomes_pc_TS_{num_patients}.joblib")
    elif timeOpt == 'TS_TR':
        kg.serialize(f"data/sphn_pc_TS_TR_{num_patients}.nt", format="nt")
        joblib.dump(df["outcome"].astype(int).to_list(), f"data/outcomes_pc_TS_TR_{num_patients}.joblib")
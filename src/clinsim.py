import click
import uuid
from string import Template
from models.phenotypes import generate_dataframe
from rdflib import ConjunctiveGraph

prefix  = """   
@prefix sphn: <http://sphn.org/> .
@prefix nvasc: <http://nvasc.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""

sphn_diagnosis_code_template = Template("""
    nvasc:$diag_id a sphn:Diagnosis ;
        rdfs:label "$diag_label" ;
        sphn:hasCode $diag_code ;
        sphn:hasSubjectPseudoIdentifier nvasc:$patient_id .
    """)
    
sphn_diagnosis_quantity_template = Template("""
    nvasc:$diag_id a sphn:Diagnosis ;
        rdfs:label "$diag_label" ;
        sphn:hasQuantity [ rdf:type sphn:Quantity ;
                            sphn:hasValue "$diag_value" ;
                            sphn:hasUnit "$diag_unit" ] ;
        sphn:hasSubjectPseudoIdentifier nvasc:$patient_id .
    """)

def gen_patient_rdf(row, kg):
    _i = row.name
    for f in row.index:
        if f in ["age", "bmi", "adjusted_size_ratio"]:
            value = None
            unit = None 
            if f == "age":
                value = row[f]
                unit = "years"
            elif  f == "bmi":
                value = row[f]
                unit = "kg/m2"
            elif  f == "adjusted_size_ratio":
                value = row[f]
                unit = "adjusted_size_ratio"
            
            rdf = sphn_diagnosis_quantity_template.substitute(diag_id=uuid.uuid4(),
                                                         diag_label=f,  diag_value=value, diag_unit=unit, patient_id=_i)
            kg.parse(data=prefix+rdf, format="turtle")
        else:    
            diag_label = f
            diag_code = row[f]
            rdf = sphn_diagnosis_code_template.substitute(diag_id=uuid.uuid4(),
                                                    diag_label=f, 
                                                    diag_code=row[f], 
                                                    patient_id=_i)
            kg.parse(data=prefix+rdf, format="turtle")

@click.command("sphn")
#@click.option('--n', required=True, type=int, help="number of samples")
@click.option('--n', default=1, type=int, help="number of samples")
def sphn_command(n):
    df = generate_dataframe(nb_sample=n)
    kg = ConjunctiveGraph()
    df.apply(gen_patient_rdf, axis=1, kg = kg)
    print(f"Generated {len(kg)} RDF triples")
    kg.serialize("sphn.ttl", format="turtle")
        
    
#@click.command("caresm")
#@click.option('--n', required=True, type=int, help="number of samples")
#@click.option('--n', default=1, type=int, help="number of samples")
#def caresm_command(n):
#    df = generate_dataframe(nb_sample=n)
#    print(df.head())
    
@click.group()
def entry_point():
    pass

entry_point.add_command(sphn_command)
#entry_point.add_command(caresm_command)
#entry_point.add_command(phenopackets_command)


if __name__ == '__main__':
    entry_point()
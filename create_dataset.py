import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from jpype import *
import jpype.imports
import os
import mowl
mowl.init_jvm("10g")

from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import PathDataset

from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat
from java.util import HashSet
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_prot_iri(org, accession):
    return f"http://mowl/protein/{org}.{accession}"

@ck.command()
@ck.option(
    '--go-file', '-ont', default='data/go-plus.owl',
    help='Ontology file (GO by default)')
@ck.option(
    "--ont", "-ont", type=ck.Choice(["mf", "bp", "cc"]), default="mf")
def main(go_file, ont):

    
    data_root = "data"
    out_dir = os.path.join(data_root, ont)

    train_pf_file = os.path.join(out_dir, 'train_data.pkl')
    valid_pf_file = os.path.join(out_dir, 'valid_data.pkl')
    test_pf_file = os.path.join(out_dir, 'test_data.pkl')
    
    train_df = pd.read_pickle(train_pf_file)
    valid_df = pd.read_pickle(valid_pf_file)

    #train, valid, test = load_and_split_interactions(data_file)
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    factory = adapter.data_factory
    dataset = PathDataset(go_file)
    train_ont = dataset.ontology
    valid_ont = manager.createOntology() 
    
    has_function_rel = adapter.create_object_property("http://mowl/has_function")
    
    # Add GO protein annotations to the GO ontology
    train_pf_axioms = HashSet()
    for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Generating training axioms"):
        protein_name = row["proteins"]
        org = row["orgs"]
        prot = adapter.create_individual(f"http://{protein_name}") # e.g. 'http://4932.YKL020C'

        go_terms = row["prop_annotations"]
        for go_term in go_terms:
            go_term = go_term.replace("GO:", "http://purl.obolibrary.org/obo/GO_")
            go_class = adapter.create_class(go_term)
            has_fn = adapter.create_object_some_values_from(has_function_rel, go_class)
            axiom = adapter.create_class_assertion(has_fn, prot)
            train_pf_axioms.add(axiom)
            
    valid_pf_axioms = HashSet()
    for i, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Generating validation axioms"):
        protein_name = row["proteins"]
        org = row["orgs"]
        prot = adapter.create_individual(f"http://{protein_name}") # e.g. 'http://4932.YKL020C'

        go_terms = row["prop_annotations"]
        for go_term in go_terms:
            go_term = go_term.replace("GO:", "http://purl.obolibrary.org/obo/GO_")
            go_class = adapter.create_class(go_term)
            has_fn = adapter.create_object_some_values_from(has_function_rel, go_class)
            axiom = adapter.create_class_assertion(has_fn, prot)
            valid_pf_axioms.add(axiom)
            
    test_pf_axioms = HashSet()
    for i, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Generating test axioms"):
        protein_name = row["proteins"]
        org = row["orgs"]
        prot = adapter.create_individual(f"http://{protein_name}") # e.g. 'http://4932.Y
        go_terms = row["prop_annotations"]
        for go_term in go_terms:
            go_term = go_term.replace("GO:", "http://purl.obolibrary.org/obo/GO_")
            go_class = adapter.create_class(go_term)
            has_fn = adapter.create_object_some_values_from(has_function_rel, go_class)
            axiom = adapter.create_class_assertion(has_fn, prot)
            test_pf_axioms.add(axiom)

    # Add axioms to the ontology
    manager.addAxioms(train_ont, train_pf_axioms)
    manager.addAxioms(valid_ont, valid_pf_axioms)
    manager.addAxioms(valid_ont, test_pf_axioms)
        
    # Save the files
    new_train_ont_file = os.path.join(out_dir, 'train.owl')
    manager.saveOntology(train_ont, OWLXMLDocumentFormat(), IRI.create('file:' + os.path.abspath(new_train_ont_file)))
    new_valid_ont_file = os.path.join(out_dir, 'valid.owl')
    manager.saveOntology(valid_ont, OWLXMLDocumentFormat(), IRI.create('file:' + os.path.abspath(new_valid_ont_file)))
    new_test_ont_file = os.path.join(out_dir, 'test.owl')
    manager.saveOntology(valid_ont, OWLXMLDocumentFormat(), IRI.create('file:' + os.path.abspath(new_test_ont_file)))


if __name__ == '__main__':
    main()
    shutdownJVM()

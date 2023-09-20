@Grapes([
    @Grab(group="org.semanticweb.elk", module="elk-owlapi", version="0.4.2"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-api", version="4.1.0"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-apibinding", version="4.1.0"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-impl", version="4.1.0"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-parsers", version="4.1.0"),
    @Grab(group="org.codehaus.gpars", module="gpars", version="1.1.0"),
    @GrabConfig(systemClassLoader=true)
])

import org.semanticweb.owlapi.model.parameters.*;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration;
import org.semanticweb.elk.reasoner.config.*;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasoner
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;
import org.semanticweb.owlapi.reasoner.structural.*;

import groovyx.gpars.GParsPool;

OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(
    new File("data/go.owl"))
OWLDataFactory dataFactory = manager.getOWLDataFactory()
OWLDataFactory fac = manager.getOWLDataFactory()
ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
ElkReasonerFactory f1 = new ElkReasonerFactory()
OWLReasoner reasoner = f1.createReasoner(ont, config)
reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

def posReg = [:].withDefault {key -> return new HashSet<String>()}
def negReg = [:].withDefault {key -> return new HashSet<String>()}

// Positive Regulation
def pr = fac.getOWLObjectProperty(
    IRI.create("http://purl.obolibrary.org/obo/RO_0002213"))
// Negative Regulation
def nr = fac.getOWLObjectProperty(
    IRI.create("http://purl.obolibrary.org/obo/RO_0002212"))


def getLabel = { term_id ->
    IRI iri = IRI.create("http://purl.obolibrary.org/obo/$term_id")
    OWLClass cl = dataFactory.getOWLClass(iri)
    for(OWLAnnotation a : EntitySearcher.getAnnotations(cl, ont, dataFactory.getRDFSLabel())) {
        OWLAnnotationValue value = a.getValue();
        if(value instanceof OWLLiteral) {
            return ((OWLLiteral) value).getLiteral();
        }
    }
    return "";
}

IRI reg_iri = IRI.create("http://purl.obolibrary.org/obo/GO_0065007")
OWLClass reg = dataFactory.getOWLClass(reg_iri)
        
GParsPool.withPool {
    ont.getClassesInSignature(true).eachParallel { cl ->
        def cls = cl.toString()
        cls = cls.substring(32, cls.length() - 1)
        def c = fac.getOWLObjectSomeValuesFrom(pr, cl)
	c = fac.getOWLObjectIntersectionOf(reg, c)
	reasoner.getEquivalentClasses(c).getEntities().each { sub ->
            def s = sub.toString()
            if (s.startsWith("<http://purl.obolibrary.org/obo/GO_")) {
                s = s.substring(32, s.length() - 1)
                posReg[s].add(cls)
            }
        }

        c = fac.getOWLObjectSomeValuesFrom(nr, cl)
	c = fac.getOWLObjectIntersectionOf(reg, c)

	reasoner.getEquivalentClasses(c).getEntities().each { sub ->
            def s = sub.toString()
            if (s.startsWith("<http://purl.obolibrary.org/obo/GO_")) {
                s = s.substring(32, s.length() - 1)
                negReg[s].add(cls)
            }
        }

    }
}


def out = new PrintWriter(new BufferedWriter(new FileWriter("data/regulations.txt")))

posReg.each { go, gos ->
    gos.each { go_id ->
	out.println("$go\t$go_id\tpos");
    }
}

negReg.each { go, gos ->
    gos.each { go_id ->
	out.println("$go\t$go_id\tneg");
    }
}

out.close();

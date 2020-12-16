package om;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.io.FileDocumentSource;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.search.EntitySearcher;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Set;


public class Get_Label_URIName {
    // place one ontology e.g., foodon-merged.owl in this folder, or multiple ontologies in this folder
    public static String onto_dir = "ontology/";
    public static String annotation_file = "foodon_class_name.json";

    public static void main(String [] args) throws OWLOntologyCreationException, IOException {
        JSONObject obj = new JSONObject();
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        File d = new File(onto_dir);
        File[] tempList = d.listFiles();
        for (int k = 0; k < tempList.length; k++) {
            File onto_file = tempList[k];

            // filter out files you don't want to process
            if (!onto_file.toString().endsWith(".owl")){
                continue;
            }

            OWLOntology ont = manager.loadOntologyFromOntologyDocument(new FileDocumentSource(onto_file));

            Set<OWLClass> classes = ont.getClassesInSignature();
            for (OWLClass e : classes) {

                String ns_e = null;
                String uri_name = null;
                String uri = e.getIRI().toString();

                for (int i = 0; i < Generate_Path_Mapping.namesapces.length; i++) {
                    uri_name = uri.replace(Generate_Path_Mapping.namesapces[i], "");
                    ns_e = uri.replace(Generate_Path_Mapping.namesapces[i], Generate_Path_Mapping.prefixes[i]);
                }

                if (uri_name.startsWith("http://")){
                    uri_name = uri_name.split("#")[1];
                }

                String label = null;
                for (OWLAnnotation a : EntitySearcher.getAnnotations(e, ont)) {
                    OWLAnnotationProperty prop = a.getProperty();
                    if (prop.toString().equals("rdfs:label")) {
                        OWLAnnotationValue val = a.getValue();
                        if (val instanceof OWLLiteral) {
                            OWLLiteral lit = (OWLLiteral) val;
                            if ((lit.hasLang() && lit.hasLang("en")) || !lit.hasLang()) {
                                label = lit.getLiteral();
                            }
                        }
                    }
                }

                JSONArray uriNameLabel = new JSONArray();
                uriNameLabel.add(uri_name);
                uriNameLabel.add(label);
                obj.put(ns_e, uriNameLabel);
            }
            System.out.println(onto_file.toString() + " done");
        }
        FileWriter file = new FileWriter(annotation_file);
        file.write(obj.toJSONString());
        file.close();
        System.out.println("all saved");
    }
}

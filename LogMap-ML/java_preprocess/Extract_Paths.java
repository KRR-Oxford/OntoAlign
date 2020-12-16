package om;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.io.FileDocumentSource;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasonerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;


public class Extract_Paths {
    public static String onto_file = "foodon-merged.owl";
    public static String path_file = "foodon_all_paths.txt";

    // include all the classes or only the classes with short prefixes
    public static boolean include_all_classes = true;

    public static OWLOntologyManager m = OWLManager.createOWLOntologyManager();
    public static OWLDataFactory df = m.getOWLDataFactory();

    public static void main(String [] args) throws OWLOntologyCreationException, IOException {
        OWLOntology onto = m.loadOntologyFromOntologyDocument(new FileDocumentSource(new File(onto_file)));
        OWLReasonerFactory reasonerFactory = new StructuralReasonerFactory();
        OWLReasoner reasoner = reasonerFactory.createReasoner(onto);
        ArrayList<OWLClass> leaves = new ArrayList<>();

        for (OWLClass c : onto.getClassesInSignature()){
            Set<OWLClass> subclasses = reasoner.getSubClasses(c, true).getFlattened();
            subclasses.remove(df.getOWLNothing());
            if(subclasses.size()==0){
                leaves.add(c);
            }
        }
        System.out.println("leaves #: " + leaves.size());

        ArrayList<String> path_strings = new ArrayList<>();
        for (OWLClass leaf : onto.getClassesInSignature()){
            Set<OWLClass> path_classes = Generate_Path_Mapping.get_path_classes(leaf, reasoner);
            ArrayList<OWLClass> path = new ArrayList<>(Arrays.asList(leaf));
            path = Generate_Path_Mapping.append_super_class(path, reasoner, path_classes);
            String path_s = Generate_Path_Mapping.path_string(path);
            if(include_all_classes){
                path_strings.add(path_s);
            }else{
                if(!path_s.contains("http://")){
                    path_strings.add(path_s);
                }
            }
        }

        File fout = new File(path_file);
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        for(String path_s : path_strings){
            bw.write(path_s + "\n");
        }
        bw.close();

    }
}

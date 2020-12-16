The ontologies' class labels/names (e.g., foodon_class_name.json) and paths (e.g., foodon_all_paths.txt) are pre-extracted.
Class labels are extracted by Get_Label_URIName.java, and paths are extracted by Extract_Paths.java.

It's suggested to build a maven project to run these java codes. These Java codes have been tested with the following dependencies:

```
 <dependencies>
        <dependency>
            <groupId>net.sourceforge.owlapi</groupId>
            <artifactId>owlapi-distribution</artifactId>
            <version>4.2.5</version>
        </dependency>
        <dependency>
            <groupId>org.codehaus.gpars</groupId>
            <artifactId>gpars</artifactId>
            <version>1.1.0</version>
        </dependency>
        <dependency>
            <groupId>net.sourceforge.owlapi</groupId>
            <artifactId>org.semanticweb.hermit</artifactId>
            <version>1.3.8.413</version>
        </dependency>
        <dependency>
            <groupId>org.semanticweb.elk</groupId>
            <artifactId>elk-owlapi</artifactId>
            <version>0.4.3</version>
        </dependency>
        <dependency>
            <groupId>ontology-services-toolkit</groupId>
            <artifactId>ontology-services-toolkit</artifactId>
            <version>1.0.0-SNAPSHOT</version>
            <scope>compile</scope>
        </dependency>
    </dependencies>
```

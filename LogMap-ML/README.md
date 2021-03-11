# LogMap-ML

This folder includes the implementation of LogMap-ML introduced in the paper ****Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**** which has been accepted by ESWC 2021.
The HeLis and FoodOn ontologies, and their partial GS is in helis_foodon.tar.gz.


### Dependence 
Our codes in this package are tested with
  1. Python 3.7
  2. Tensorflow 1.13.1
  3. gensim 3.8.0
  4. [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star)
  5. [LogMap v3.0](https://github.com/ernestojimenezruiz/logmap-matcher)
  6. [AML](https://github.com/AgreementMakerLight/AML-Project) (Optional)


### Startup

#### Pre-process #1: Running the original system
Run LogMap, get its output mappings, overlapping mappings and anchor mappings, by

```java -jar target/logmap-matcher-3.0.jar MATCHER file:/xx/helis_v1.00.owl file:/xx/foodon-merged.owl output/ true```

Note LogMap has been updated to V4.0 which now uses OWL API 4. 
No functional changes are made from V3.0 to V4.0, and thus the ML extension should still be able to work for LogMap V4.0.
 
#### Pre-process #2: Ontology Embedding
You can either use the word2vec embedding by gensim ([download](https://drive.google.com/file/d/1rm9uJEKG25PJ79zxbZUWuaUroWeoWbFR/view?usp=sharing)), 
or the ontology tailored [OWL2Vec\* embedding](https://github.com/KRR-Oxford/OWL2Vec-Star). 
The to-be-aligned ontologies can be set with their own embedding models or be set with one common embedding model.

#### Pre-process #3: Path and Class Name Extraction
We use Java OWL API to pre-extract all the paths and class names of the to-be-aligned ontologies. 
They are saved as intermediate files (xx_all_paths.txt and xx_class_name.json) by the two java programs under java_preprocess/.
The name and path files of HeLis and FoodOn are already in helis_foodon.tar.gz.

#### Step #1: Sample
```python sample.py```

See the parameter "help" and comment inside the program for different settings. 
The branch conflicts which are manually set for higher quality seed mappings are set inside the program.
It will output mappings_train.txt and mappings_valid.txt.

#### Step #2: Train, valid and predict
```python train_valid.py```

```python predict_candidates.py```

Note the candidate mappings should be pre-extracted by some ontology alignment systems or 
downloaded from OAEI as we did in the experiment for some settings. 
One direct candidate source is the overlapping mappings by LogMap.

#### Step #3: Evaluate
Calculate the recall w.r.t. the GS, and sample a number of mappings for annotation, by:

```python evaluate.py```

It will output a file with a part of the mappings for human annotation. 
The annotation is done by appending "true" or "false" to each mapping (see annotation example in evaluate.py).
With the manual annotation and the GS, the precision and recall can be approximated by:

```python approximate_precision_recall.py```

Please see Eq. (2)(3)(4) in the paper for how the precision and recall approximation works.
For more accurate approximate, it is suggested to annotate and use the mappings of at least three systems to approximate the GS. 
Besides the original LogMap and LogMap-ML, you can also consider [AML](https://github.com/AgreementMakerLight/AML-Project) as well.

========================================

> Note: this is a preliminary implementation. We are making it more "end-to-end".

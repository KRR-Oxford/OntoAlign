# LogMap-ML

This folder includes the implementation of LogMap-ML introduced in the paper ****Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision****.
The HeLis and FoodOn ontologies, and their partial GS, which are adopted for the evaluation in the paper, are under **data/**.
Note the HeLis ontology adopted has been pre-processed by transforming instances into classes.


### Dependence 
Our codes in this package require: 
  1. Python 3.8.5
  2. Tensorflow 2.4.1
  3. gensim 4.0.1
  4. OWLready2 0.29
  5. [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star)
  6. [LogMap v3.0](https://github.com/ernestojimenezruiz/logmap-matcher)
  7. [AML](https://github.com/AgreementMakerLight/AML-Project) (Optional)


### Startup

### Pre-process #1: Run the original system
Download [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher), build by Maven, run:

```java -jar target/logmap-matcher-4.0.jar MATCHER file:/xx/helis_v1.00.owl file:/xx/foodon-merged.owl logmap_output/ true```

This leads to LogMap output mappings, overlapping mappings and anchor mappings. 
For the input ontologies, ``file:`` is necessary; for the output directory, ``/`` is necessary.
Note LogMap has been updated to V4.0 which now uses OWL API 4. No functional changes are made from V3.0 to V4.0, and thus LogMap-ML should still be able to work for LogMap V4.0.
 
### Pre-process #2: Embedding Models
You can either use the word2vec embedding by gensim (The one trained by English Wikipedia articles in 2018 [download](https://drive.google.com/file/d/1rm9uJEKG25PJ79zxbZUWuaUroWeoWbFR/view?usp=sharing)), 
or the ontology tailored [OWL2Vec\* embedding](https://github.com/KRR-Oxford/OWL2Vec-Star). 
The to-be-aligned ontologies can use their own embedding models or use one common embedding model.

### Pre-process #3: Class Name and Path Extraction
``python name_path.py --onto_file data/xx.owl --name_file data/xx_class_name.json --path_file data/xx_all_paths.txt``

This is to extract the name information and path information for each class in an ontology. 
It should be executed separately for the to-be-aligned ontologies.

### Step #1: Sample
```python sample.py --anchor_mapping_file logmap_output/logmap_anchors.txt --left_class_name_file data/xx_class_name.json --left_path_file data/xx_all_paths.txt --right_class_name_file data/xx_class_name.json --right_path_file data/xx_all_paths.txt --train_file data/mappings_train.txt --valid_file data/mappings_valid.txt```

It outputs mappings_train.txt and mappings_valid.txt.
The branch conflicts which are manually set for higher quality seed mappings are set inside the program via the variable ``branch_conflicts``.
See "help" and comments inside the program for more settings. 

### Step #2: Train, Valid and Predict
```python train_valid.py --left_w2v_dir dir/word2vec_gensim --right_w2v_dir dir/word2vec_gensim --train_path_file data/mappings_train.txt --valid_path_file data/mappings_valid.txt```

Path type can be set via ``--path_type`` (use the isolated class, the path from the class to the root, or URI name plus the class and the parent); Networks and path encoding can be set via the variables ``nn_types`` and ``encoder_types``; see more settings in "Help".

```python predict_candidates.py --candidate_file logmap_output/logmap_overestimation.txt --left_w2v_dir dir/word2vec_gensim --right_w2v_dir dir/word2vec_gensim```

Note the candidate mappings should be pre-extracted, usually with a high recall. We use the overlapping mappings by LogMap.
``predict_candidates.py`` by default outputs mapping scores in predict_score.txt.

### Step #3: Evaluate
Calculate the recall w.r.t. the GS, and sample a number of mappings for annotation, by:

```python evaluate.py --threshold 0.65 --anchor_file logmap_output/logmap_anchors.txt```

It will output a file with a part of the mappings for human annotation. 
The annotation is done by appending "true" or "false" to each mapping (see annotation example in evaluate.py).
With the manual annotation and the GS, the precision and recall can be approximated by:

```python approximate_precision_recall.py```

Please see Eq. (2)(3)(4) in the paper for how the precision and recall approximation works.
For more accurate approximate, it is suggested to annotate and use the mappings of at least three systems to approximate the GS. 
Besides the original LogMap and LogMap-ML, you can also consider [AML](https://github.com/AgreementMakerLight/AML-Project) as well.

========================================

> Update on 24 Aug: move to support TF 2.x and Gensim 4.0.1. 

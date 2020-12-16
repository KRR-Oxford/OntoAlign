# LogMap-ML

This folder includes the implementation for the paper ****Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision****.
The HeLis and FoodOn ontologies, and their partial GS is in helis_foodon.tar.gz.

### Dependence 
Our codes in this package are tested with
  1. Python 3.7
  2. Tensorflow 1.13.1
  3. gensim 3.8.0
  4. [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star)


### Startup

##### Pre-process #0: Running the original system
Run e.g., LogMap, get its output mappings, overlapping mappings and anchor mappings.

##### Pre-process #1: Ontology Embedding
You can either use some word2vec embedding by gensim ([download](https://drive.google.com/file/d/1rm9uJEKG25PJ79zxbZUWuaUroWeoWbFR/view?usp=sharing)), or the pre-trained OWL2Vec\* embedding. 
Note the to-be-aligned ontologies can be set with their own embeddings or be set with one embedding. 
OWL2Vec\* is able embed multiple ontologies into one language model.

##### Pre-process #2: Path and Class Name Extraction
We use Java OWL API to pre-extract all the paths and names of the to-be-aligned ontologies. See java_preprocess/.

##### Step #1: Sample
```python sample.py```

See the help for different settings.

##### Step #2: Train, valid and predict
```python train_valid.py```

```python predict_candidates.py```

Note the candidate mappings should be pre-extracted, or use the overlapping mappings by LogMap.

##### Step #3: Evaluate
Calculate the recall w.r.t. the GS, and sample a number of mappings for annotation, by:

```python evaluate.py```

Annotate the sampled mapping by appending "true" or "false", and then approximate the precision and recall by:

```python approximate_precision_recall.py```

Note it is suggested to annotate and use the mappings of at least three systems to approximate the GS. 
Thus run, sample and annotate for LogMap and AML as well.

## Ablation study

One of the main contributions of this thesis is also going to be an ablation study of different models and their
components. The goal is to understand the importance of different components of the models and how they affect the
performance of the model. The ablation study will be done on two datasets, heterogeneous and homogeneous. Addtionally,
we will test the importance of the following properties:

- adding contextual **embeddings** of articles to the model,

- the **GNN backbone**, where we primarily hypothesise that the novel transformer and GATv2 backbones will outperform
  the LightGCN baseline and SAGE state-of-the-art model,
- the **loss function**, where we will test Bayesian Personalized Ranking (BPR) and Binary Cross-Entropy with
  Logits (BCEWLL),
- adding research trends to the model, where we hypothesise that the model exploiting research trend data will
  outperform the model without research trends,
- breaking down the embeddings into **uniform periodical averages** to simulate the change in the author's research
  interest,
  We derive the following combinations of the components to test:

| Model              | Backbone    | Graph type    | Loss   | Embeddings                  | Research trends? | 
|--------------------|-------------|---------------|--------|-----------------------------|------------------|
| [A](model_A.ipynb) | LightGCN    | Homogeneous   | BPR    | /                           | No               | 
| [B](model_B.ipynb) | SAGE        | Homogeneous   | BPR    | average                     | No               | 
| [C](model_C.ipynb) | GATv2       | Homogeneous   | BPR    | average                     | No               | 
| [D](model_D.ipynb) | Transformer | Homogeneous   | BPR    | average                     | No               | 
| [E](model_E.ipynb) | GATv2       | Homogeneous   | BCEWLL | average                     | No               | 
| [F](model_F.ipynb) | Transformer | Homogeneous   | BCEWLL | average                     | No               | 
| [G](model_G.ipynb) | GATv2       | Heterogeneous | BPR    | average                     | No               | 
| [H](model_G.ipynb) | Transformer | Heterogeneous | BPR    | average                     | No               | 
| [I](model_H.ipynb) | GATv2       | Homogeneous   | BPR    | weighted periodical average | No               | 
| [J](model_J.ipynb) | GATv2       | Homogeneous   | BPR    | average                     | Yes              | 

### Additional notebooks

- [build_homogeneous_dataset.ipynb](build_homogeneous_dataset.ipynb) - notebook for building the homogeneous dataset.
- [build_heterogeneous_dataset.ipynb](build_heterogeneous_dataset.ipynb) - notebook for building the heterogeneous
  dataset.
- [results_interpretation.ipynb](results_interpretation.ipynb) - notebook for interpreting the results of the models.
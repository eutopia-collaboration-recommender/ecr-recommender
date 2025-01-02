## Ablation study

One of the main contributions of this thesis is also going to be an ablation study of different models and their
components. The goal is to understand the importance of different components of the models and how they affect the
performance of the model. The ablation study will be done on two datasets, heterogeneous and homogeneous. Addtionally,
we will test the importance of the following properties:

- adding contextual **embeddings** of articles to the model,
- the **GNN backbone**, where we primarily hypothesise that the novel transformer and GATv2 backbones will outperform
  the
  LightGCN baseline and SAGE state-of-the-art model,
- the **loss function**, where we will test Bayesian Personalized Ranking (BPR) and Binary Cross-Entropy with
  Logits (BCEWLL),
- adding research trends to the model, where we hypothesise that the model exploiting research trend data will
  outperform the model without research trends,
- the **lead author** of the paper, where we hypothesise that the model with the lead author data point will outperform
  the model without the data point.

We derive the following combinations of the components to test:

| Model              | Backbone           | Graph type    | Loss   | Embeddings? | Research trends? | Lead author? |
|--------------------|--------------------|---------------|--------|-------------|------------------|--------------|
| [A](model_A.ipynb) | LightGCN           | Homogeneous   | BPR    | No          | No               | No           |
| [B](model_B.ipynb) | SAGE               | Homogeneous   | BPR    | Yes         | No               | No           |
| [C](model_C.ipynb) | GATv2              | Homogeneous   | BPR    | Yes         | No               | No           |
| [D](model_D.ipynb) | Transformer        | Homogeneous   | BPR    | Yes         | No               | No           |
| [E](model_E.ipynb) | GATv2              | Homogeneous   | BCEWLL | Yes         | No               | No           |
| [F](model_F.ipynb) | Transformer        | Homogeneous   | BCEWLL | Yes         | No               | No           |
| [G](model_G.ipynb) | GATv2/Transformer? | Homogeneous   | BPR    | Yes         | Yes              | No           |
| [H](model_H.ipynb) | GATv2              | Heterogeneous | BPR    | Yes         | No               | No           |
| [I](model_I.ipynb) | Transformer        | Heterogeneous | BPR    | Yes         | No               | No           |
| [J](model_J.ipynb) | GATv2/Transformer? | Heterogeneous | BPR    | Yes         | No               | Yes          |

### Additional notebooks

- [build_homogeneous_dataset.ipynb](build_homogeneous_dataset.ipynb) - notebook for building the homogeneous dataset.
- [build_heterogeneous_dataset.ipynb](build_heterogeneous_dataset.ipynb) - notebook for building the heterogeneous
  dataset.
- [results_interpretation.ipynb](results_interpretation.ipynb) - notebook for interpreting the results of the models.
# Molecule Retrieval with Natural Language Queries

**Abstract**

*This project was carried out as a final project-challenge of the ALTEGRAD (Advance Learning for Text and Graph Data) course of the MVA Master of the ENS Paris-Saclay. The objective of this work was to use the machine-learning machinery to retrieve molecules, represented as graphs, from natural language queries. For the challenge, a list of molecules and a text query were given for this purpose. Two encoders were implemented, a graph encoder based on a GCN (Graph Convolution Network) and GAT (Graph Attention Networks), and a sci-BERT encoder, as the text encoder. Then, a contrastive learning approach was taken to learn a mutual representation space mapping together the text and molecule representations.*

All the implementations described in *report.pdf* are in this repo:

- *src/dataloader*: A class is defined to easily manipulate the data, text, and graph representation of molecules.
- *src/enhance_drop_sub*: The code to perform the two data augmentation described, node dropping, and subgraph with random walk.
- *src/hyper_tuning*: The script of the hyperparameter tuning loop.
- *src/main*: The main training loop.
- *src/Model*: Models are defined here, text and graphs encoders.
- src/Model-Adv*: Another version of the model, graph econder with residuals connection.

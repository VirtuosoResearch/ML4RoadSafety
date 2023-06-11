### Generate Node Embeddings

1. We use `networkx` and `karateclub` packages. Install them as follows:

```
pip install networkx karateclub
```

2. Create folder named `node2vec` and `deepwalk`:

```
mkdir node2vec
mkdir deepwalk
```

2. Use `generate_node2vec_embedding.py` and `generate_deepwalk_embedding.py` to generate node2vec and deepwalk embeddings:

```
python generate_node2vec_embedding.py --state_name DE
python generate_node2vec_embedding.py --state_name IA
python generate_node2vec_embedding.py --state_name IL
python generate_node2vec_embedding.py --state_name MA
python generate_node2vec_embedding.py --state_name MD
python generate_node2vec_embedding.py --state_name MN
python generate_node2vec_embedding.py --state_name MT
python generate_node2vec_embedding.py --state_name NV

python generate_deepwalk_embedding.py --state_name DE
python generate_deepwalk_embedding.py --state_name IA
python generate_deepwalk_embedding.py --state_name IL
python generate_deepwalk_embedding.py --state_name MA
python generate_deepwalk_embedding.py --state_name MD
python generate_deepwalk_embedding.py --state_name MN
python generate_deepwalk_embedding.py --state_name MT
python generate_deepwalk_embedding.py --state_name NV
```
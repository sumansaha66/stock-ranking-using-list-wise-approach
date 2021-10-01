
from stellargraph import StellarGraph
import networkx as nx
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

def relation_node2vec(relation_encoding_org,num_random_walks,len_random_walk,p_val,q_val,n2vemb_size):
    relation_encoding_2d_weighted=relation_encoding_org.sum(axis=2) # weighted 2d graph with sum
    relation_encoding_2d_unweighted=(relation_encoding_2d_weighted>0).astype(int) # unweighted 2d graph
    G_nx=nx.from_numpy_array(relation_encoding_2d_unweighted) # loading as networkx graph
    G = StellarGraph.from_networkx(G_nx)
    print(G.info())
    # Corpus generation using random walks
    rw = BiasedRandomWalk(G)
    walks = rw.run(
            nodes=list(G.nodes()),  # root nodes
            length=len_random_walk,  # maximum length of a random walk
            n=num_random_walks,  # number of random walks per root node
            p=p_val,  # Defines (unormalised) probability, 1/p, of returning to source node
            q=q_val,  # Defines (unormalised) probability, 1/q, for moving away from source node
            )
    print("Number of random walks: {}".format(len(walks))) # total number for all nodes in the graph
    # Representation Learning using Word2Vec
    str_walks = [[str(n) for n in walk] for walk in walks]
    # size: Dimensionality of the output
    # window: Maximum distance between the current and predicted word within a sentence
    # min_count : Ignores all words with total frequency lower than this
    # sg : {0, 1}, Training algorithm: 1 for skip-gram; otherwise CBOW.
    # iter : Number of iterations (epochs) over the corpus.
    # workers : Use these many worker threads to train the model (=faster training with multicore machines)
    model = Word2Vec(str_walks, size=n2vemb_size, window=5, min_count=0, sg=1, workers=2, iter=1)
    # Retrieve node embeddings and corresponding subjects
    node_embeddings = (model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality
    print(node_embeddings.shape)
    return node_embeddings
    



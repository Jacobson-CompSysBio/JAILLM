# import necessary packages
import networkx as nx


def convert_to_nx(filename, graph_type = nx.Graph):

    """
    Function to convert a graph from a file to a networkx graph object

    Parameters:
        filename (str): path to the file containing the graph
        graph_type (networkx.Graph): type of graph to be created
    
    Returns:
        graph (networkx.Graph): networkx graph object
    """

    graph = nx.read_edgelist(filename,
                             create_using=graph_type,
                             nodetype=str,
                             data=(('weight', float),))
    return graph



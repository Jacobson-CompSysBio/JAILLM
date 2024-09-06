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

def nx_to_text(G):
    """
    Function to convert a networkx graph object to a list of nodes and edges

    Parameters:
        G (networkx.Graph): networkx graph object
    
    Returns:
        nodes (list): list of nodes
        edges (list): list of edges
    """

    nodes = []
    for x in G.nodes():
        nodes.append(x)

    # collect edges and weights
    edges = []
    for u,v in G.edges():
        edges.append("("+str(u)+","+str(v)+") with weight " + str(G.get_edge_data(u,v)['weight']))

    return nodes, edges

def format_chat(row,
                input_col, 
                output_col,
                pipeline_name):

    """
    Function to format a row of data into a chat template for llama input

    Parameters:
        row (dict): dictionary containing the row data
        input_col (str): column name for the input
        output_col (str): column name for the output
        pipeline (llama.Pipeline): llama pipeline object
    
    Returns:
        row (dict): dictionary containing the chat template
    """

    row_json = [{'role': 'user', 'content': row[input_col]},
                {'role': 'assistant', 'content': row[output_col]}]
    row["text"] = pipeline_name.tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

def format_network_chat(row,
                        nodes, 
                        edges, 
                        question,
                        pipeline_name):

    """
    Function to format a node and edge list into a chat template for llama input

    Parameters:
        nodes (list): list of nodes
        edges (list): list of edges
        question (str): question to be asked
        pipeline (llama.Pipeline): llama pipeline object

    Returns:
        row (dict): dictionary containing the chat template
    """

    row_json = [{'role': 'user',
                 'content': 'In and undirected weighted graph, (i,j) means that node i and node j are connected with an undirected, weighted edge. The nodes are: {} and the edges are: {}\n Is there a cycle in this graph?'},
                {'role': 'assistant', 'content': 'yes'}]
    row["text"] = pipeline_name.tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

def tokenize_data(examples,
                  pipeline_name):
    
    """
    Function to preprocess the data for training on next-character prediction

    Parameters:
        examples (dict): dictionary containing the text data
        pipeline (llama.Pipeline): llama pipeline object
    
    Returns:
        tokenized_data (dict): dictionary containing the tokenized data
    """

    tokenized_data = pipeline_name.tokenizer(text=examples['text'],
                               padding='max_length', 
                               truncation=True, 
                               max_length=1024)
    
    labels = tokenized_data['input_ids'].copy()
    
    for i in range(len(labels)):
        if labels[i][-1] != pipeline_name.tokenizer.pad_token_id:
            labels[i] = labels[i][1:] + [pipeline_name.tokenizer.pad_token_id]
        else:
            labels[i] = labels[i][1:] + [pipeline_name.tokenizer.pad_token_id]

    labels = [[pipeline_name.tokenizer.pad_token_id if x == pipeline_name.tokenizer.pad_token_id else x for x in y] for y in labels]
    tokenized_data['labels'] = labels
    
    return tokenized_data

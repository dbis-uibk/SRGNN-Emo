import os.path

import pandas as pd
import numpy as np
from requests import get
from tqdm import tqdm
from collections import defaultdict
import scipy
import pickle
import argparse
import json


def flatten(l):
    return [item for sublist in l for item in sublist]


def symmetric_normalize(coo_matrix):
    rowsum = np.array(coo_matrix.sum(axis=1))
    rowsum[rowsum == 0] = 1
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(coo_matrix).dot(d_mat_inv_sqrt).tocoo()


def build_tag_graph(dict_filepath, item_id_map_dict, num_node, cutoff=2, blacklist=None, verbose=False):
    tag_item_map = defaultdict(list)
    item_tag_map = defaultdict(list)

    with open(dict_filepath, 'r') as f:
        header = next(f)
        for i, line in tqdm(enumerate(f), disable=not verbose):
            item, tags = line.split("\t")
            if item in item_id_map_dict.keys():
                item_id = item_id_map_dict[item]
                tags_dict = json.loads(tags[:-1].replace("\'", "\""))
                sorted_tags = dict(sorted(tags_dict.items(), key=lambda x: x[1], reverse=True))
                if blacklist is not None:
                    sorted_tags = {k: v for k, v in sorted_tags.items() if k not in blacklist}
                for g in list(sorted_tags.keys())[:cutoff]:
                    tag_item_map[g].append(item_id)
                    item_tag_map[item_id].append(g)

    tag_graph = np.zeros((num_node, num_node))
    for item_id, tags in tqdm(item_tag_map.items(), disable=not verbose):
        tag_items = flatten([tag_item_map[t] for t in tags])
        for t_item in tag_items:
            tag_graph[item_id, t_item] += 1
            #tag_graph[t_item, item_id] += 1

    tag_graph = scipy.sparse.coo_matrix(tag_graph)

    sparsity = 1.0 - (tag_graph.nnz / float(tag_graph.shape[0] * tag_graph.shape[1]))
    print("tag_graph")
    print("\tsparsity:\t", sparsity)
    print("\tmax weight:\t", tag_graph.max())
    print("\tnon-zero edges:\t", tag_graph.nnz)

    return tag_graph


def build_genre_graph(tf_idf_filepath, item_id_map_dict, num_node, threshold=0.5, verbose=False):
    coarse_grained_genres = ["rock", "pop", "electronic", "metal", "alternativerock", "indierock"] # see Kowald et al., https://github.com/pmuellner/supporttheunderground
    genre_item_map = defaultdict(list)
    item_genre_map = defaultdict(list)

    with open(tf_idf_filepath, 'r') as f:
        header = next(f)
        genre_names = np.array(list(header.split("\t"))[1:])
        for i, line in tqdm(enumerate(f), disable=not verbose):
            line = line.split("\t")
            item = line[0]
            genre_values = np.array(line[1:]).astype(float)
            # filter out coarse-grained genres
            genre_values = np.array([v if g not in coarse_grained_genres else 0 for g, v in zip(genre_names, genre_values)])
            if item in item_id_map_dict.keys():
                item_id = item_id_map_dict[item]
                item_genres = genre_names[np.nonzero(genre_values > threshold)]
                for g in item_genres:
                    genre_item_map[g].append(item_id)
                    item_genre_map[item_id].append(g)

    genre_graph = np.zeros((num_node, num_node))
    for item_id, genres in tqdm(item_genre_map.items(), disable=not verbose):
        genre_items = flatten([genre_item_map[g] for g in genres])
        for g_item in genre_items:
            genre_graph[item_id, g_item] += 1

    genre_graph = scipy.sparse.coo_matrix(genre_graph)

    sparsity = 1.0 - (genre_graph.nnz / float(genre_graph.shape[0] * genre_graph.shape[1]))
    print("genre_graph")
    print("\tsparsity:\t", sparsity)
    print("\tmax weight:\t", genre_graph.max())
    print("\tnon-zero edges:\t", genre_graph.nnz)

    return genre_graph


def build_session_graph(opt, timestamp_filepath, item_id_map_dict, num_node, verbose=False):
    print("loading session data...")
    if opt.dataset == 'sample':
        df = pd.read_csv(timestamp_filepath, sep='\t', nrows=10000000)
        #df = pd.read_csv(timestamp_filepath, sep='\t')
    else:
        df = pd.read_csv(timestamp_filepath, sep='\t')
        #df = pd.read_csv(timestamp_filepath, sep='\t', nrows=10000000)

    # create sessions
    print("create sessions...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    cond1 = df.timestamp - df.timestamp.shift(1) > pd.Timedelta(30, 'm')
    cond2 = df.user_id != df.user_id.shift(1)
    df['session_id'] = (cond1 | cond2).cumsum()

    df['timestamp'] = df['timestamp'].values.astype(np.int64) // 10 ** 9

    session_item_map = df.groupby('session_id')['track_id'].apply(list).to_dict()
    item_session_map = df.groupby('track_id')['session_id'].apply(list).to_dict()

    del df

    session_graph = np.zeros((num_node, num_node))
    for item, sessions in tqdm(item_session_map.items(), disable=not verbose):
        if item in item_id_map_dict.keys():
            item_id = item_id_map_dict[item]

            session_items = flatten([session_item_map[s] for s in sessions])
            session_items = [item_id_map_dict[item] for item in session_items if item in item_id_map_dict.keys()]
            session_items = [s_item for s_item in session_items if s_item != item_id]  # remove self
            for s_item in session_items:
                session_graph[item_id, s_item] += 1
    
    session_graph = scipy.sparse.coo_matrix(session_graph)

    sparsity = 1.0 - (session_graph.nnz / float(session_graph.shape[0] * session_graph.shape[1]))
    print("session_graph")
    print("\tsparsity:\t", sparsity)
    print("\tmax weight:\t", session_graph.max())
    print("\tnon-zero edges:\t", session_graph.nnz)

    return session_graph


def get_all_genres(tf_idf_filepath, threshold=1.5, verbose=False):
    all_genre_names = set()
    with open(tf_idf_filepath, 'r') as f:
        header = next(f)
        genre_names = np.array(list(header.split("\t"))[1:])
        for i, line in tqdm(enumerate(f), disable=not verbose):
            line = line.split("\t")
            genre_values = np.array(line[1:]).astype(float)
            all_genre_names.update(genre_names[np.nonzero(genre_values > threshold)])
    return all_genre_names


def get_tag_idf_scores(dict_filepath, item_id_map_dict):
    n_documents = 0
    tag_counter = defaultdict(int)
    with open(dict_filepath, 'r') as f:
        header = next(f)
        for i, line in enumerate(f):
            item, tags = line.split("\t")
            if item in item_id_map_dict.keys():
                n_documents += 1
                tags_dict = json.loads(tags[:-1].replace("\'", "\""))
                for g in tags_dict.keys():
                    tag_counter[g] += 1

    idf_scores = {k: np.log(n_documents / v) for k, v in tag_counter.items()}
    # sort
    idf_scores = dict(sorted(idf_scores.items(), key=lambda x: x[1], reverse=False))
    # ["rock", "pop", "alternative", "favorites", "indie", "love", "alternative rock"]
    return idf_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='sample', help='sample/music4all-onion')
    parser.add_argument('--verbose', '-v', default=0)
    opt = parser.parse_args()

    if opt.dataset == 'music4all-onion' or opt.dataset == 'm4a':
        opt.dataset = 'music4all-onion'
        num_node = 109267
    else:
        num_node = 10000

    print(opt)
    item_id_map_dict = pickle.load(open('../data/' + opt.dataset + '/item_id_map_dict.pkl', 'rb'))
    all_genres = get_all_genres('../data/' + opt.dataset + '/id_genres_tf-idf.tsv', threshold=0.5, verbose=opt.verbose)
    
    blacklist_tags = list(all_genres) + ["rock", "pop", "alternative", "favorites", "indie", "love", "alternative rock"]

    print("build graphs...")
    session_graph = build_session_graph(opt, '../data/music4all-onion/userid_trackid_timestamp.tsv', item_id_map_dict, num_node, verbose=opt.verbose)
    tag_graph = build_tag_graph('../data/' + opt.dataset + '/id_tags_dict.tsv', item_id_map_dict, num_node, cutoff=2, blacklist=all_genres, verbose=opt.verbose)
    genre_graph = build_genre_graph('../data/' + opt.dataset + '/id_genres_tf-idf.tsv', item_id_map_dict, num_node, threshold=0.5, verbose=opt.verbose)
    
    # add self loops
    session_graph = session_graph + scipy.sparse.eye(session_graph.shape[0])
    tag_graph = tag_graph + scipy.sparse.eye(tag_graph.shape[0])
    genre_graph = genre_graph + scipy.sparse.eye(genre_graph.shape[0])

    # row normalize the weights in the individual matrices
    session_graph_normalized = symmetric_normalize(session_graph)
    tag_graph_normalized = symmetric_normalize(tag_graph)
    genre_graph_normalized = symmetric_normalize(genre_graph)

    all_graphs = [session_graph_normalized, tag_graph_normalized, genre_graph_normalized]
    edge_index_src = []
    edge_index_dst = []
    edge_weight = []
    edge_type = []
    for g_id, g in enumerate(all_graphs):
        edge_index_src.append(g.row)
        edge_index_dst.append(g.col)
        edge_weight.append(g.data)
        edge_type.append(np.ones_like(g.data) * g_id)

    edge_index_src = np.concatenate(edge_index_src)
    edge_index_dst = np.concatenate(edge_index_dst)
    edge_weight = np.concatenate(edge_weight)
    edge_type = np.concatenate(edge_type)
    edge_index = np.stack([edge_index_src, edge_index_dst], axis=0)

    data = (edge_index, edge_weight, edge_type)

    print("write to file...")
    with open('../data/' + opt.dataset + '/graph.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("finished")

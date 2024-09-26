def get_hyperparameters(opt, dataset='sample', rep='musicnn'):
    if opt.rep == "musicnn":
        opt.input_dim = 50
        opt.projection = 0
    elif opt.rep == "jukebox":
        opt.input_dim = 4800
        opt.projection = 1
    elif opt.rep == "maest":
        opt.input_dim = 768
        opt.projection = 1

    if dataset == 'music4all-onion' or dataset == 'm4a':
        if rep == "musicnn":
            opt.num_node = 109267
            opt.batch_size = 60000
            opt.neighbors = 40
            opt.folds = 10  

            opt.layers = 2
            opt.de_1 = 0.05
            opt.df_1 = 0.15
            opt.de_2 = 0.1
            opt.df_2 = 0.05

            opt.gnn_dropout = 0.0
            opt.dropout = 0.2

            opt.alpha = 0.01 
            opt.beta = 0.07 

            opt.clusters = 10
            opt.confidence_threshold = 0.5
            opt.k = 5

        elif rep == "jukebox":
            opt.num_node = 109267
            opt.batch_size = 40000
            opt.neighbors = 20
            opt.folds = 10  

            opt.layers = 2
            opt.de_1 = 0.05
            opt.df_1 = 0.15
            opt.de_2 = 0.1
            opt.df_2 = 0.05

            opt.gnn_dropout = 0.0
            opt.dropout = 0.2

            opt.alpha = 0.01  
            opt.beta = 0.07 

            opt.clusters = 10
            opt.confidence_threshold = 0.5
            opt.k = 5

        elif rep == "maest":
            opt.num_node = 109267
            opt.batch_size = 40000
            opt.neighbors = 20
            opt.folds = 10  

            opt.layers = 2
            opt.de_1 = 0.05
            opt.df_1 = 0.15
            opt.de_2 = 0.1
            opt.df_2 = 0.05

            opt.gnn_dropout = 0.0
            opt.dropout = 0.2

            opt.alpha = 0.01 
            opt.beta = 0.07 

            opt.clusters = 10
            opt.confidence_threshold = 0.5
            opt.k = 5

    return opt
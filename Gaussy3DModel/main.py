from GaussyModel import GaussyModel


def main():
    '''
        Programa Que recibe X=Tengo tiempo y Y=Tengo dinero => Z=Voy al cine
    '''

    X = [
        [
            0, # Median
            6, # Desv_std
            None # Gaussy_values
        ],
        [
            15, 7, None
        ],
        [
            30, 6, None
        ]
    ]

    Y = [
        [
            0, # Median
            6, # Desv_std
            None # Gaussy_values
        ],
        [
            15, 7, None
        ],
        [
            30, 6, None
        ]
    ]


    fussy_rules = [
        [
            -0.1, # p
            3,    # q
            0,    # r
            X[0], # X
            Y[0]  # Y
        ],
        [
            0.001, # p
            1,    # q
            0,    # r
            X[0], # X
            Y[1]  # Y
        ],
        [
            0.08, # p
            1,    # q
            0,    # r
            X[0], # X
            Y[2]  # Y
        ],
        [
            -0.2, # p
            3,    # q
            0,    # r
            X[1], # X
            Y[0]  # Y
        ],
        [
            0.002, # p
            1,    # q
            0,    # r
            X[1], # X
            Y[1]  # Y
        ],
        [
            0.07, # p
            1,    # q
            0,    # r
            X[1], # X
            Y[2]  # Y
        ],
        [
            -0.3, # p
            3,    # q
            0,    # r
            X[2], # X
            Y[0]  # Y
        ],
        [
            0.003, # p
            1,    # q
            0,    # r
            X[2], # X
            Y[1]  # Y
        ],
        [
            0.06, # p
            1,    # q
            0,    # r
            X[2], # X
            Y[2]  # Y
        ]
    ]
    
    tmpModel = GaussyModel(
                x_elements = X,
                y_elements = Y,
                rules=fussy_rules, 
                step=1,
                step_range=[0,3],
                debuglevel=0)
    network = tmpModel.fit()
    print(network)


if __name__ == "__main__":
    main()
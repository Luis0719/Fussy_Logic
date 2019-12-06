from GaussyModel import GaussyModel


def main():
    '''
        Programa Que recibe X=Tengo tiempo y Y=Tengo dinero => Z=Voy al cine
    '''

    X = [
        {
            'median': 1,
            'desv_std': 6,
            'gaussy_values': None
        },
        {
            'median': 14,
            'desv_std': 7,
            'gaussy_values': None
        },
        {
            'median': 29,
            'desv_std': 6,
            'gaussy_values': None
        }
    ]

    Y = [
        {
            'median': 0,
            'desv_std': 6,
            'gaussy_values': None
        },
        {
            'median': 15,
            'desv_std': 7,
            'gaussy_values': None
        },
        {
            'median': 30,
            'desv_std': 6,
            'gaussy_values': None
        }
    ]


    fussy_rules = [
        {
            'p': -0.1,
            'q': 3,
            'r': 0,
            'x': X[0],
            'y': Y[0]
        },
        {
            'p': 0.001,
            'q': 1,
            'r': 0,
            'x': X[0],
            'y': Y[1]
        },
        {
            'p': 0.08,
            'q': 1,
            'r': 0,
            'x': X[0],
            'y': Y[2]
        },
        {
            'p': -0.2,
            'q': 3,
            'r': 0,
            'x': X[1],
            'y': Y[0]
        },
        {
            'p': 0.002,
            'q': 1,
            'r': 0,
            'x': X[1],
            'y': Y[1]
        },
        {
            'p': 0.07,
            'q': 1,
            'r': 0,
            'x': X[1],
            'y': Y[2]
        },
        {
            'p': -0.3,
            'q': 3,
            'r': 0,
            'x': X[2],
            'y': Y[0]
        },
        {
            'p': 0.003,
            'q': 1,
            'r': 0,
            'x': X[2],
            'y': Y[1]
        },
        {
            'p': 0.06,
            'q': 1,
            'r': 0,
            'x': X[2],
            'y': Y[2]
        }
    ]
    
    tmpModel = GaussyModel(
                x_elements = X,
                y_elements = Y,
                rules=fussy_rules, 
                step=1,
                step_range=[0,50],
                debuglevel=5)
    network = tmpModel.fit()
    print(network)


if __name__ == "__main__":
    main()
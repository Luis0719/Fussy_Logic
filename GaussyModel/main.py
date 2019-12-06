from GaussyModel import GaussyModel

def main():
    temperatures = [
        {
            'median': 0,
            'desv_std': 6,
            'p': -0.1,
            'q': 3,
            'gaussy_values': None
        },
        {
            'median': 15,
            'desv_std': 7,
            'p': 0.001,
            'q': 1,
            'gaussy_values': None
        },
        {
            'median': 30,
            'desv_std': 6,
            'p': 0.08,
            'q': 1,
            'gaussy_values': None
        }
    ]
    tmpModel = GaussyModel(
                    variables=temperatures, 
                    x_step=0.1,
                    bottom_limit=0,
                    top_limit=500)
    tmpModel.fit()


if __name__ == "__main__":
    main()
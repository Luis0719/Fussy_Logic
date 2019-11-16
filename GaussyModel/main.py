from GaussyModel import GaussyModel

def main():
    temperatures = [
        {
            'median': 0,
            'desv_std': 6,
            'p': 0.04,
            'q': 1,
            'gaussy_values': None
        },
        {
            'median': 18,
            'desv_std': 6,
            'p': 0.05,
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
                    bottom_limit=-200,
                    top_limit=500)
    tmpModel.fit()


if __name__ == "__main__":
    main()
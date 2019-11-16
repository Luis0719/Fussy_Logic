from GaussyModel import GaussyModel

def main():
    temperatures = [
        {
            'median': 0,
            'desv_std': 6,
            'p': 0.04,
            'q': 0
        },
        {
            'median': 18,
            'desv_std': 6,
            'p': 0.05,
            'q': 0
        },
        {
            'median': 30,
            'desv_std': 6,
            'p': 0.08,
            'q': 0
        }
    ]
    tmpModel = GaussyModel(
                    variables=temperatures, 
                    x_step=0.1)
    tmpModel.fit()


if __name__ == "__main__":
    main()
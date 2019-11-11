from GaussyModel import GaussyModel

def main():
    temperatures = [
        {
            'bottom': -24,
            'top': 24,
            'desv_std': 9,
            'p': 1,
            'q': 1
        },
        {
            'bottom': 19,
            'top': 36,
            'desv_std': 3,
            'p': 1,
            'q': 1
        },
        {
            'bottom': 30,
            'top': 50,
            'desv_std': 4,
            'p': 1,
            'q': 1
        }
    ]
    tmpModel = GaussyModel(
                    temperatures=temperatures, 
                    x_step=0.1)
    tmpModel.fit()


if __name__ == "__main__":
    main()
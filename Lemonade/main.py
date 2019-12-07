from GeneticAlgorith import PSmodel

def main():
    target = [
        [1,3,4,4,3],
        [3,4,7,9,6],
        [2,7,10,9,8],
        [3,6,9,8,7],
        [3,4,6,9,8]
    ]
    resolutions=[51, 51, 1000, 1000, 51]

    model = PSmodel(population_size=200,
        cromosome_size=39,
        target_values=target,
        generations=1,
        competidors_percentage=0.05,
        mutation_percentage=0.01,
        elitism=False,
        resolutions=resolutions,
        graph_generations=False,
        bottom_limit=0,
        top_limit=5,
        step=1,
        debuglevel=5
    )
    model.fit()

if __name__ == "__main__":
    main()

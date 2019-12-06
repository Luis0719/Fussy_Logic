from GeneticAlgorith import PSmodel

def main():
    target = [
        [1,1,1,1,1],
        [1,2,2,2,2],
        [1,2,3,3,3],
        [1,2,2,2,2],
        [1,1,1,1,1],
    ]
    resolutions=[51, 51, 1000, 1000, 51]

    model = PSmodel(population_size=1000,
        cromosome_size=39,
        target_values=target,
        generations=10,
        competidors_percentage=0.05,
        mutation_percentage=0.01,
        elitism=False,
        resolutions=resolutions,
        graph_generations=True,
        x_bottom_limit=0,
        x_top_limit=5,
        x_step=1,
        debuglevel=5
    )
    model.fit()

if __name__ == "__main__":
    main()

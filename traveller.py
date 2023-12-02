import numpy as np
from copy import deepcopy


class Path(object):
    def __init__(self, path, graph):
        assert type(path) is list
        self._path = path
        self._graph = graph
        self._num_city = len(graph)
        self._value = self._get_value()

    def path(self):
        return self._path

    def value(self):
        return self._value

    def _get_value(self):
        '''Returns the value of genetic code
           Value is a number (integer or floating point).'''
        assert type(self._path) is list

        if self._path[0] != self._path[-1] or len(self._path) != self._num_city + 1:
            return 1e-9

        cost = 0
        for idx, city in enumerate(self._path):
            if idx == len(self._path) - 1:
                break
            next_city = self._path[idx + 1]
            cost += self._graph[city][next_city]

        return 1/cost

    def __gt__(self, other):
        return self._value > other.value()

    def __add__(self, other):
        """
        'Crossover method' for genetic search. It should return a new genetic
         code that is the 'mix' of father and mother.
        """
        assert type(other) is Path

        half_path1 = deepcopy(self._path[:len(self._path)//2])
        half_path2 = deepcopy(other.path())
        half_path2 = list(filter(lambda city: city not in half_path1, half_path2))

        return Path(half_path1 + half_path2, self._graph)

    def mutate(self):
        middle = len(self._path)//2
        pos_1 = np.random.randint(0, middle)
        pos_2 = np.random.randint(middle, len(self._path))
        self._path[pos_1], self._path[pos_2] = self._path[pos_2], self._path[pos_1]
        self._value = self._get_value()

class GeneticSearch(object):
    def __init__(self, graph):
        assert type(graph) is dict
        self._graph = graph
        self._num_city = len(graph)

    def _get_random_path(self):
        """
        generate random genetic code
        """
        cities = list(self._graph)
        city_probs = np.ones(self._num_city) / self._num_city
        path = np.random.choice(cities, self._num_city, p=city_probs, replace=False)
        end_city = np.random.choice(cities, 1, p=city_probs)
        return path.tolist() + end_city.tolist()

    def _get_random_parents(self, population):
        population_values = [path.value() for path in population]
        total_values = sum(population_values)
        '''probabilities of each individual, each represent how likely that
           individual is chosen to become a parent'''
        population_probs = [value/total_values for value in population_values]
        return np.random.choice(population, 2, p=population_probs, replace=False)

    def _population_expander(self, population_size, mutation_chance):
        def expander(population):
            new_generation = []
            for _ in range(population_size):
                parents = self._get_random_parents(population)
                father, mother = parents.tolist()
                child = father + mother

                '''random if a child is likely to be mutated'''
                is_mutant = np.random.choice([True, False], 1, p=[mutation_chance, 1-mutation_chance])[0]
                if is_mutant:
                    child.mutate()
                new_generation.append(child)

            population += new_generation #add new generation to the current population
            population_values = [path.value() for path in population]
            total_values = sum(population_values)
            population_probs = [value/total_values for value in population_values]

            '''randomly choose individuals to fit new population'''
            population = np.random.choice(population, population_size, p=population_probs, replace=False)
            return population.tolist()

        return expander

    def search(self, num_generation=100, population_size=100, mutation_chance=0.1, patience=5):
        assert type(num_generation) is int and type(population_size) is int
        assert (0 <= mutation_chance <= 1) and (1 <= num_generation) and (1 <= patience)
        population = []
        expander = self._population_expander(population_size, mutation_chance)

        for _ in range(population_size):
            p = self._get_random_path()
            population.append(Path(p, self._graph))
        best_individual = max(population)

        count_generation = 0
        patience_current = 0
        while count_generation < num_generation and patience_current < patience:
            population = expander(population)
            next_gen_best_individual = max(population)
            if next_gen_best_individual > best_individual:
                best_individual = next_gen_best_individual
                patience_current = 0
            else:
                patience_current += 1
            count_generation += 1
        return best_individual

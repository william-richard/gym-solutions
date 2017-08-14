import gym
import numpy
import random
import math
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, env, genes):
        self._env = env
        self._genes = numpy.copy(genes)
        self._fitness = None

    def get_fitness(self, render=False):
        if not self._fitness:
            fitnesses = []
            for i in range(50):
                observation = self._env.reset()
                total_reward = 0.0
                for t in range(200):
                    if render:
                        self._env.render()
                        #print(observation)
                    #print(self._genes)
                    mean = numpy.average(observation, weights=self._genes)
                    #print(mean)
                    if mean >= 0:
                        action = 0
                    else:
                        action = 1
                    observation, reward, done, info = self._env.step(action)
                    total_reward += reward
                    #print(reward, info)
                    if done:
                        break
                fitnesses.append(total_reward)
            self._fitness = numpy.mean(fitnesses)
            #print(f"Calculated fitness of {self._fitness}")

        return self._fitness

    def breed(self, other_individual):
        return Individual(self._env, (self._genes + other_individual._genes) / 2.0)

    def mutate(self, mutation_rate):
        for i, gene in enumerate(self._genes):
            if random.random() < mutation_rate:
                new_gene = random.uniform(-1.0, 1.0)
                self._genes[i] = new_gene

    def reset_fitness(self):
        self._fitness = None

class Population:
    def __init__(self, env, size = 20, initialize_size = 0):
        self._env = env
        self._size = size
        self._population = []
        for i in range(initialize_size):
            # TODO don't hard code this size
            #genes = [math.floor(random.random()) + .0001 for g in range(4)]
            genes = [random.uniform(-1.0, 1.0) for g in range(4)]
            new_individual = Individual(env, genes)
            self._population.append(new_individual)

    def __len__(self):
        return len(self._population)

    def size(self):
        return self._size

    def get_most_fit(self, n=1):
        self._population.sort(key=lambda i: i.get_fitness(), reverse=True)
        return self._population[0:n]
        #most_fit = random.choice(self._population)
        #for individual in self._population:
        #    if individual.get_fitness() > most_fit.get_fitness():
        #        most_fit = individual
        #return most_fit

    def get_average_fitness(self):
        sum = 0.0
        for individual in self._population:
            sum += individual.get_fitness()
        return numpy.round(sum / len(self._population), 2)

    def add_individual(self, individual):
        if len(self._population) > self._size:
            raise ValueError(f"Population has {len(self._population)} members, but max size of {self._size}")
        self._population.append(individual)

    def tournament(self, tournament_size):
        tournament_winners = random.choices(self._population, k=tournament_size)
        tournament_pop = Population(self._env, tournament_size, 0)
        for winner in tournament_winners:
            tournament_pop.add_individual(winner)
        return tournament_pop.get_most_fit(1)[0]

    def mutate(self, mutation_rate):
        for individual in self._population:
            individual.mutate(mutation_rate)

class GA:
    def __init__(self, env, population_size, mutation_rate = 0.015, tournament_size = 5, new_percentage = .05, keep_percentage = 0.1):
        self._env = env
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._tournament_size = tournament_size
        self._new_percentage = new_percentage
        self._keep_percentage = keep_percentage

    def evolve_population(self, old_population):
        new_population = Population(self._env, self._population_size, math.floor(self._population_size * self._new_percentage))

        keepers = old_population.get_most_fit(math.floor(self._population_size * self._keep_percentage))
        for keeper in keepers:
            keeper.reset_fitness()
            new_population.add_individual(keeper)
            #most_fit = old_population.get_most_fit()
            #most_fit.reset_fitness()
            #new_population.add_individual(most_fit)

        while len(new_population) < new_population.size():
            ind_1 = old_population.tournament(self._tournament_size)
            ind_2 = old_population.tournament(self._tournament_size)
            new_ind = ind_1.breed(ind_2)
            new_population.add_individual(new_ind)

        new_population.mutate(self._mutation_rate)

        return new_population

def main():

    env = gym.make('CartPole-v0')

    population_size = 100
    population = Population(env, population_size, population_size)
    ga = GA(env, population_size, tournament_size = 30, mutation_rate = 0.05, new_percentage = 0.00, keep_percentage = 0.1)

    most_fit_history = []
    avg_fit_history = []

    for generation in range(100):
        print(f"Starting generation {generation}")
        most_fit = population.get_most_fit()[0]
        print(f"Current most fit individual has fitness {numpy.round(most_fit.get_fitness(), 2)} and population has average fitness {population.get_average_fitness()}")
        most_fit_history.append(most_fit.get_fitness())
        avg_fit_history.append(population.get_average_fitness())

        population = ga.evolve_population(population)

        plt.clf()
        plt.plot(most_fit_history, label='Most fit')
        plt.plot(avg_fit_history, label='Average fitness')

        plt.xlabel('generation')
        plt.ylabel('fitness score')

        plt.legend()

        plt.pause(0.01)

        #render_most_fit = Individual(env, most_fit._genes)
        #render_most_fit.reset_fitness()
        #print(f"Rendered fitness {render_most_fit.get_fitness(True)}")

    plt.show()


if __name__ == "__main__":
    main()

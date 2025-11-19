from random import random
import time
import csv
import pandas as pd
import os
import numpy as np
class GeneticAlgorithm():
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate, eliticism_rate, iterations, selection_type=1, crossover_type=1):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.eliticism_rate = eliticism_rate
        self.crossover_type = crossover_type
        self.selection_type = selection_type
        self.iterations = iterations
        self.weights, self.values, self.things, self.numbers = self.get_data_from_csv()
        self.objects_names = dict(zip(self.numbers, self.things))
        self.objects_values= dict(zip(self.numbers, self.values))

    def save_to_csv(self):
        header = ['Pc', 'Pm', 'N', 'T', 'Run',
                  'Best_chromosome', 'Best_value', 'Best_items',
                  'Worst_value', 'Average_population_history', 'Time_exec']

    def get_data_from_csv(self):
        df = pd.read_csv('algorytm_plecakowy.csv')
        return df['Waga'].values.tolist(), df['Wartosc'].values.tolist(), df['Nazwa'].values.tolist(), df['Numer'].values.tolist()

    def fitness_function(self, chromosome):
        weight = sum([self.weights[i] * chromosome[i] for i in range(len(chromosome))])
        value = sum([self.values[i] * chromosome[i] for i in range(len(chromosome))])
        return value if weight <= 35 else 0

    def init_population(self):
        return np.random.randint(2, size=(self.population_size, self.chromosome_length))

    def crossover(self, chromosome1, chromosome2):
        crossover_point = np.random.randint(1, self.chromosome_length)
        new_chromosome1 = np.concatenate([chromosome1[:crossover_point], chromosome2[crossover_point:]])
        new_chromosome2 = np.concatenate([chromosome2[:crossover_point], chromosome1[crossover_point:]])
        return new_chromosome1, new_chromosome2

    def dual_crossover(self, chromosome1, chromosome2):
        child1 = chromosome1.deepcopy()
        child2 = chromosome1.deepcopy()
        crossover_point = np.random.randint(1, self.chromosome_length - 2)
        second_crossover_point = np.random.randint(crossover_point + 1, self.chromosome_length - 1)
        child1[crossover_point:second_crossover_point] = chromosome2[crossover_point:second_crossover_point]
        child2[crossover_point:second_crossover_point] = chromosome1[crossover_point:second_crossover_point]
        return child1, child2

    def mutate(self, chromosome):
        chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def elitism(self, population):
        fitness = np.array([self.fitness_function(chromosome) for chromosome in population])
        sorted_population = population[np.argsort(fitness)[::-1]]
        elite_size = int(self.eliticism_rate * self.population_size)
        return sorted_population[:elite_size]

    def select_roulette(self, population):
        fitness = np.array([self.fitness_function(chromosome) for chromosome in population])
        total = np.sum(fitness)
        if total == 0:
            return population[np.random.randint(len(population))]
        probability = fitness / total
        return population[np.random.choice(len(population), p=probability)]

    def tournament_selection(self, population):
        tournament_contestant_ind1 = random.randint(0, len(population) - 1)
        tournament_contestant_ind2 = random.randint(0, len(population) - 1)

        fitness_value1 = self.fitness_function(population[tournament_contestant_ind1])
        fitness_value2 = self.fitness_function(population[tournament_contestant_ind2])

        if fitness_value1 > fitness_value2:
            return population[tournament_contestant_ind1]
        else:
            return population[tournament_contestant_ind2]

    def run(self):
        start = time.time()
        population = self.init_population()
        average_population_history = []
        results = []
        for epoch in range(self.iterations):
            fitness = np.array([self.fitness_function(chromosome) for chromosome in population])
            elite_chromosomes = self.elitism(population)
            new_population = []
            new_population.extend(elite_chromosomes)
            new_population_size_to_pool = self.population_size - len(elite_chromosomes)
            for j in range(new_population_size_to_pool // 2):
                if self.selection_type == 1:
                    chromosome1 = self.select_roulette(population)
                    chromosome2 = self.select_roulette(population)
                else:
                    chromosome1= self.tournament_selection(population)
                    chromosome2 = self.tournament_selection(population)

                if np.random.rand() < self.crossover_rate:
                    if self.crossover_type == 1:
                        chromosome1, chromosome2 = self.crossover(chromosome1, chromosome2)
                    else:
                        chromosome1, chromosome2 = self.dual_crossover(chromosome1, chromosome2)

                chromosome1 = self.mutate(chromosome1)
                chromosome2 = self.mutate(chromosome2)
                new_population.append(chromosome1)
                new_population.append(chromosome2)

            population = np.array(new_population[:self.population_size])
            population_for_mean = population[np.argmax(fitness)]
            population_the_worst = population[np.argmin(fitness)]
            chosen_objects = [i for i in range(len(population_for_mean)) if population_for_mean[i] == 1]
            values_objects = [self.objects_values[i + 1] for i in chosen_objects]
            names_objects = [self.objects_names[i + 1] for i in chosen_objects]
            avg_population = sum(values_objects) / len(values_objects)
            average_population_history.append(avg_population)
            results.append({
                'Pc': self.crossover_rate,
                'Pm': self.mutation_rate,
                'N': self.population_size,
                'T': epoch + 1,
                'Best_chromosome': population_for_mean,
                'Best_value': sum(values_objects),
                'Best_items': names_objects,
                'Worst_value': population_the_worst,
                'Average_population_history': avg_population,
                'Time_exec': time.time() - start
            })
        fitness = np.array([self.fitness_function(chromosome) for chromosome in population])
        chromosome = population[np.argmax(fitness)]
        end = time.time()
        chosen_objects = [i for i in range(len(chromosome)) if chromosome[i] == 1]
        values_objects = [self.objects_values[i + 1] for i in chosen_objects]
        names_objects = [self.objects_names[i + 1] for i in chosen_objects]
        time_exec = end - start
        results[-1]['Time_exec'] = time_exec
        print(f"Rozwiązanie: {chromosome}, wartość fitness function: {self.fitness_function(chromosome)}")
        print(f"Wybrano rzeczy o numerach: {chosen_objects}")
        print(f"Wybrano rzeczy o nazwach: {names_objects}")
        print(f"Wybrano rzeczy o wartościach: {values_objects}")
        print(f"Wartosc wszystkich rzeczy: {sum(values_objects)}")
        print(f"Czas wykonania: {end - start:.6f} s")
        #print(f"Średnia wartość rozwiązania w populacji w każdej iteracji: {list(enumerate(average_population_history, start=1))}")
        df2 = pd.DataFrame(results)
        filename = "results.csv"
        i = 1
        while os.path.exists(filename):
            filename = f'results{i}.csv'
            i += 1

        df2.to_csv(filename, index=False)
        return chromosome, sum(values_objects), names_objects, time_exec, average_population_history

    def analyse_results(self, data):
        all_files = os.listdir(".")
        result_files = [f for f in all_files if f.startswith('final_results') and f.endswith('.csv')]
        output = []
        for file in result_files:
            df = pd.read_csv(file)
            data = df.tolist()
            max_value = max(data)
            min_value = min(data)
            mean = sum(data)/len(data)
            if len(data) % 2 == 0:
                median = sum(data[len(data)/2-1:len(data)/2])
            else:
                median = data[len(data)/2]
            #for population?
            standard_deviation =  np.std(data)
            output.append(max_value, min_value, mean, median, standard_deviation)
            print(f"max wartość: {max_value}, min wartość: {min_value}, mean: {mean}, median: {median}, standard deviation: {standard_deviation}")
        return output

def save_best_results():
    data = []
    all_files = os.listdir(".")
    result_files = [f for f in all_files if f.startswith('results') and f.endswith('.csv')]
    if not result_files:
        print("Nie znaleziono plików results*.csv")
    else:
        for file in result_files:
            df = pd.read_csv(file)
            data.append(df.iloc[-1].tolist())
    df2 = pd.DataFrame(data)
    filename = "final_results.csv"
    i = 1
    while os.path.exists(filename):
        filename = f'final_results{i}.csv'
        i += 1
    df2.to_csv(filename, index=False)

if __name__ == '__main__':
    df1 = pd.read_csv('parameters.csv')
    chromoseome_length = 26
    eliticism_rate = 0.25
    for i in range(5):
        ga = GeneticAlgorithm(df1['N'].values[0], 26, df1['pm'].values[0], df1['pc'].values[0], eliticism_rate, df1['T'].values[0], crossover_type=df1['crossover_type'].values[0], selection_type=df1['selection_type'].values[0])
        x = ga.run()
    save_best_results()
    print(x)


'''
Dla każdego uruchomienia zapisz:
najlepsze rozwiązanie (wartość plecaka i jego zawartość),
najgorsze rozwiązanie,
średnią wartość rozwiązania w populacji w każdej iteracji,
czas wykonania.
Zapisz wyniki w tabelach lub plikach CSV (aby można było je potem łatwo porównać i narysować wykresy).
'''
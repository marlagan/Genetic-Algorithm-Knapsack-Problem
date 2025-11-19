from random import random
import time
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import random

for_combinations = []

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
        child1 = chromosome1.copy()
        child2 = chromosome1.copy()
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

def analyse_results():
    all_files = os.listdir(".")
    result_files = [f for f in all_files if f.startswith('final_results') and f.endswith('.csv')]
    summary_rows = []
    for file in result_files:
        df = pd.read_csv(file)
        best_values = df['Best_value'].values.tolist()
        max_value = np.max(best_values)
        min_value = np.min(best_values)
        mean_value = np.mean(best_values)
        median_value = np.median(best_values)
        std_value = np.std(best_values)
        Pc = df["Pc"].iloc[0]
        Pm = df["Pm"].iloc[0]
        N = df["N"].iloc[0]
        T = df["T"].iloc[0]
        summary_rows.append({
            "File": file,
            "Pc": Pc,
            "Pm": Pm,
            "N": N,
            "T": T,
            "Max": max_value,
            "Min": min_value,
            "Mean": mean_value,
            "Median": median_value,
            "Std": std_value
        })
        print(f" Pc={Pc}, Pm={Pm}, N={N}, T={T}")
        print(f" Max={max_value}")
        print(f" Min={min_value}")
        print(f" Mean={mean_value}")
        print(f" Median={median_value}")
        print(f" Std={std_value}")

    summary_df = pd.DataFrame(summary_rows)
    i = 1
    filename_csv = "result_for_chart.csv"
    filename_table = "result_for_chart.txt"
    while os.path.exists(filename_csv) or os.path.exists(filename_table):
        filename_csv = f"result_for_chart{i}.csv"
        filename_table = f"result_for_chart{i}.txt"
        i += 1
    summary_df.to_csv(filename_csv, index=False)
    with open(filename_table, "w", encoding="utf-8") as f:
        f.write(summary_df.to_string(index=False))

    return summary_df

def save_best_results():
    data = []
    all_files = os.listdir(".")
    result_files = [f for f in all_files if f.startswith('results') and f.endswith('.csv')]
    if not result_files:
        print("Nie znaleziono plików results*.csv")
        return
    headers = [
        "Pc",
        "Pm",
        "N",
        "T",
        "Best_chromosome",
        "Best_value",
        "Best_items",
        "Worst_chromosome",
        "Worst_value",
        "Time_exec"
    ]
    best_output = 0
    best_file = None
    for file in result_files:
        df = pd.read_csv(file)
        if float(df.iloc[-1]['Best_value']) > best_output:
            best_output = df.iloc[-1]['Best_value']
            best_file = file
        last_row = df.iloc[-1].tolist()
        data.append(last_row)
    df2 = pd.DataFrame(data, columns=headers)
    filename = "final_results.csv"
    filename_table = f"final_results.txt"
    i = 1
    while os.path.exists(filename):
        filename = f"final_results{i}.csv"
        filename_table = f"final_results{i}.txt"
        i += 1
    df2.to_csv(filename, index=False)
    with open(filename_table, "w", encoding="utf-8") as f:
        f.write(df2.to_string(index=False))
    create_chart_in_time(best_file)
    return df2

def create_chart_in_time(filename):
    df = pd.read_csv(filename)
    last_row = df.iloc[-1]
    plt.plot(range(1, last_row['T'] + 1), df['Best_value'])
    plt.xlabel("Iteracja")
    plt.ylabel("Najlepszy wynik")
    plt.title("Zmiany najlepszego rozwiązania w czasie")
    plt.grid(True)
    chart_filename = "chart.png"
    i = 1
    while os.path.exists(chart_filename):
        chart_filename = f"chart{i}.png"
        i += 1
    plt.savefig(chart_filename)
    plt.show()

def create_chart_param_comparison():
    df = pd.read_csv("result_for_chart.csv")
    for param in ['Pc', 'Pm', 'N', 'T']:
        plt.figure(figsize=(8, 6))
        for value in sorted(df[param].unique()):
            y_values = df[df[param] == value]['Max'].values
            plt.scatter([value] * len(y_values), y_values, label=f"{param}={value}")
        plt.xlabel(param)
        plt.ylabel("Najlepszy wynik (Max)")
        plt.title(f"Wpływ {param} na najlepszy wynik")
        plt.grid(True)
        plt.tight_layout()
        save_prefix = 'parameters_comparison'
        i = 1
        filename = f"{save_prefix}.png"
        while os.path.exists(filename):
            filename = f"{save_prefix}{i}.png"
            i += 1
        plt.savefig(filename)
        plt.show()

def generate_charts_combinations():
    combinations = [[1, 1], [1, 2], [2, 1], [2, 2]]
    method_name = {
        (1, 1): "Ruletka 1-punktowe krzyżowanie",
        (1, 2): "Ruletka 2-punktowe krzyżowanie",
        (2, 1): "Turniej 1-punktowe krzyżowanie",
        (2, 2): "Turniej 2-punktowe krzyżowanie"
    }
    grouped_results = defaultdict(list)
    for row in for_combinations:
        key = (row[0], row[1], row[2], row[3])
        grouped_results[key].append(row)
    for params, results in grouped_results.items():
        plt.figure(figsize=(8, 6))
        methods = []
        values = []
        for i, row in enumerate(results):
            crossover_type, selection_type = combinations[i]
            method = method_name[(crossover_type, selection_type)]
            methods.append(method)
            values.append(row[5])
        plt.bar(methods, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        plt.ylabel("Najlepszy wynik", fontsize=12)
        plt.xlabel("Metoda selekcji i krzyżowania", fontsize=12)
        plt.title(f"Porównanie metod dla parametrów:\nPc={params[0]}, Pm={params[1]}, N={params[2]}, T={params[3]}",
                  fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        filename = f"comparison_Pc{params[0]}_Pm{params[1]}_N{params[2]}_T{params[3]}.png"
        i = 1
        while os.path.exists(filename):
            filename = f"comparison_Pc{params[0]}_Pm{params[1]}_N{params[2]}_T{params[3]}_{i}.png"
            i += 1
        plt.savefig(filename)
        plt.show()

if __name__ == '__main__':
    df1 = pd.read_csv('parameters.csv')
    eliticism_rate = 0.25
    chromosome_length = 26
    combinations = [[1, 1], [1, 2], [2, 1], [2, 2]]
    for z in combinations:
        for x in range(len(df1)):
            for y in range(5):
                #crossover_type = int(df1.at[df1.index[x], 'crossover_type'])
                #election_type = int(df1.at[df1.index[x], 'selection_type'])
                N = int(df1.at[df1.index[x], 'N'])
                pm = float(df1.at[df1.index[x], 'pm'])
                pc = float(df1.at[df1.index[x], 'pc'])
                T = int(df1.at[df1.index[x], 'T'])
                ga = GeneticAlgorithm(N, chromosome_length, pm, pc, eliticism_rate, T, crossover_type=z[0],
                                      selection_type=z[1])
                o = ga.run()
                print(o)
            df2 = save_best_results()
            for_combinations.append(df2.iloc[0])
            for file in os.listdir("."):
                if file.startswith("results"):
                    os.remove(file)
        analyse_results()
        create_chart_param_comparison()
    generate_charts_combinations()


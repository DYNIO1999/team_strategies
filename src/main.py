import numpy as np
import time

class ACO_Knapsack:
    def __init__(self, num_items, values, weights, max_weight, num_ants, num_iterations, decay, alpha, beta):
        self.num_items = num_items
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        self.pheromone = np.ones(num_items)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def local_search(self, initial_solution):
        current_solution = initial_solution[:]
        current_value, current_weight = self.evaluate_solution(current_solution)
        improved = True

        while improved:
            improved = False
            for i in range(self.num_items):
                if current_solution[i] == 0:
                    new_solution = current_solution[:]
                    new_solution[i] = 1
                    new_value, new_weight = self.evaluate_solution(new_solution)
                    if new_weight <= self.max_weight and new_value > current_value:
                        current_solution = new_solution
                        current_value = new_value
                        improved = True
                else:
                    new_solution = current_solution[:]
                    new_solution[i] = 0
                    new_value, new_weight = self.evaluate_solution(new_solution)
                    if new_value > current_value:
                        current_solution = new_solution
                        current_value = new_value
                        improved = True

                    for j in range(self.num_items):
                        if new_solution[j] == 0:
                            swap_solution = new_solution[:]
                            swap_solution[i] = 0
                            swap_solution[j] = 1
                            swap_value, swap_weight = self.evaluate_solution(swap_solution)
                            if swap_weight <= self.max_weight and swap_value > current_value:
                                current_solution = swap_solution
                                current_value = swap_value
                                improved = True

        return current_solution, current_value

    def run(self):
        current_best_value = 0
        current_best_solution = None
        for _ in range(self.num_iterations):
            solutions = self.construct_solutions()
            self.update_pheromones(solutions)
            for solution in solutions:
                value, weight = self.evaluate_solution(solution)
                if weight <= self.max_weight:
                    refined_solution, refined_value = self.local_search(solution)
                    if refined_value > current_best_value:
                        current_best_value = refined_value
                        current_best_solution = refined_solution
        return current_best_solution, current_best_value

    def construct_solutions(self):
        solutions = []
        for _ in range(self.num_ants):
            solution = []
            for i in range(self.num_items):
                item_prob = pow(self.pheromone[i], self.alpha) * pow(self.values[i]/self.weights[i], self.beta)
                if np.random.rand() < item_prob:
                    solution.append(1)
                else:
                    solution.append(0)
            solutions.append(solution)
        return solutions

    def update_pheromones(self, solutions):
        self.pheromone *= (1 - self.decay)  # Pheromone evaporation
        for solution in solutions:
            value, weight = self.evaluate_solution(solution)
            if weight <= self.max_weight:
                for i, included in enumerate(solution):
                    if included:
                        self.pheromone[i] += value / weight  # Update pheromone

    def evaluate_solution(self, solution):
        value = sum([v * s for v, s in zip(self.values, solution)])
        weight = sum([w * s for w, s in zip(self.weights, solution)])
        return value, weight


def run_and_print_result(knapsack_solver, number_of_runs):

    for _ in range(number_of_runs):
        start_time = time.time()
        best_solution, best_value = knapsack_solver.run()
        end_time = time.time()
        execution_time = end_time - start_time

        print("Best solution: ", best_solution)
        print("Best value: ", best_value)
        print(f"Execution time: {execution_time} seconds")


def tunning_process_grid_search(num_items, values, weights, max_weight):

    alphas = [0.5, 1, 2, 3, 4, 5]
    betas = [1, 2, 3, 4, 5]
    decays = [0.1, 0.2, 0.3, 0.4, 0.5]
    num_ants = [num_items, num_items*2]
    num_iterations = [100, 500, 1000]

    for alpha in alphas:
        for beta in betas:
            for decay in decays:
                for ants in num_ants:
                    for iterations in num_iterations:
                        aco = ACO_Knapsack(num_items, values, weights, max_weight, ants, iterations, decay, alpha, beta)
                        solution, value = aco.run()
                        print(
                            f"Alpha: {alpha}, "
                            f"Beta: {beta}, "
                            f"Decay: {decay}, "
                            f"Ants: {ants}, "
                            f"Iterations: {iterations}, "
                            f"Value: {value}")


def main():

    num_items_P01 = 10
    values_P01 = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
    weights_P01 = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
    max_weight_P01 = 165

    num_items_P02 = 5
    values_P02 = np.array([24, 13, 23, 15, 16])
    weights_P02 = np.array([12, 7, 11, 8, 9])
    max_weight_P02 = 26

    num_items_P06 = 7
    values_P06 = np.array([442, 525, 511, 593, 546, 564, 617])
    weights_P06 = np.array([41, 50, 49, 59, 55, 57, 60])
    max_weight_P06 = 170

    num_items_P07 = 15
    values_P07 = np.array([135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240])
    weights_P07 = np.array([70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120])
    max_weight_P07 = 750

    num_items_P08 = 24
    values_P08 = np.array([
        825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457,
        1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538,
        675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261
    ])
    weights_P08 = np.array([
        382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150,
        823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111,
        323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684
    ])
    max_weight_P08 = 6404180

    number_of_runs = 3

    # #optimal 309
    # aco_knapsack_p01 = ACO_Knapsack(num_items=num_items_P01, values=values_P01, weights=weights_P01,
    #                                 max_weight=max_weight_P01, num_ants=num_items_P01*2, num_iterations=1000, decay=0.5,
    #                                 alpha=1,
    #                                 beta=1.5)
    #
    # run_and_print_result(aco_knapsack_p01, number_of_runs)
    #
    # #optimal 51
    # aco_knapsack_p02 = ACO_Knapsack(num_items=num_items_P02, values=values_P02, weights=weights_P02,
    #                                 max_weight=max_weight_P02, num_ants=num_items_P02*2, num_iterations=1000, decay=0.5, alpha=2,
    #                                 beta=1)
    #
    # run_and_print_result(aco_knapsack_p02, number_of_runs)
    #
    # #optimal 1735
    # aco_knapsack_p06 = ACO_Knapsack(num_items=num_items_P06, values=values_P06, weights=weights_P06,
    #                                 max_weight=max_weight_P06, num_ants=num_items_P06*2, num_iterations=1000, decay=0.5, alpha=1,
    #                                 beta=1)
    #
    # run_and_print_result(aco_knapsack_p06, number_of_runs)

    #optimal 1458
    aco_knapsack_p07 = ACO_Knapsack(num_items=num_items_P07, values=values_P07, weights=weights_P07,
                                    max_weight=max_weight_P07, num_ants=num_items_P07, num_iterations=200, decay=0.3, alpha=0.5,
                                    beta=3)

    tunning_process_grid_search(num_items_P07, values_P07, weights_P07, max_weight_P07)


    #run_and_print_result(aco_knapsack_p07, number_of_runs)


    #optimal 13549094
    aco_knapsack_p08 = ACO_Knapsack(num_items=num_items_P08, values=values_P08, weights=weights_P08,
                                    max_weight=max_weight_P08, num_ants=num_items_P08, num_iterations=200, decay=0.3, alpha=0.5,
                                    beta=3)

    #run_and_print_result(aco_knapsack_p07, number_of_runs)



if __name__ == "__main__":
    main()



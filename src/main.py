import numpy as np
import time
import matplotlib.pyplot as plt


class ACO_Knapsack:
    def __init__(self, num_items, values, weights, max_weight, num_ants, num_iterations, decay, alpha, beta, optimal_value):
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
        self.optimal_value =optimal_value

    def local_search(self, solution):
        improved = True
        while improved:
            improved = False
            current_solution = solution.copy()
            current_value, current_weight = self.evaluate_solution(current_solution)

            for i in range(self.num_items):
                if current_solution[i] == 0:
                    if current_weight + self.weights[i] <= self.max_weight:
                        test_solution = current_solution.copy()
                        test_solution[i] = 1
                        test_value, test_weight = self.evaluate_solution(test_solution)
                        if test_value > current_value:
                            solution = test_solution.copy()
                            current_value = test_value
                            current_weight = test_weight
                            improved = True
                else:
                    test_solution = current_solution.copy()
                    test_solution[i] = 0
                    removed_weight = current_weight - self.weights[i]
                    for j in range(self.num_items):
                        if j != i and test_solution[j] == 0 and removed_weight + self.weights[j] <= self.max_weight:
                            swap_solution = test_solution.copy()
                            swap_solution[j] = 1
                            swap_value, swap_weight = self.evaluate_solution(swap_solution)
                            if swap_value > current_value:
                                solution = swap_solution.copy()
                                current_value = swap_value
                                current_weight = swap_weight
                                improved = True
                                break

        return solution, current_value, current_weight

    def draw_weight_to_values(self, weights, values):

        plt.figure(figsize=(12,  6))

        x_axis = [i for i in range(len(weights))]

        plt.plot(x_axis, [self.max_weight for _ in range(len(weights))],  label='max weight', linewidth=2)
        plt.plot(x_axis, [self.optimal_value for _ in range(len(weights))],  label='optimal value', linewidth=2)
        plt.plot(x_axis, weights, label='weights')
        plt.plot(x_axis, values, label='values')

        plt.title('Weights and Values')
        plt.xlabel('Ants*Iterations')
        plt.ylabel('weights and values')

        plt.legend()

        plt.show()

    def run(self):
        current_best_value = 0
        current_best_solution = None

        weights_list = []
        values_list = []

        for _ in range(self.num_iterations):
            solutions = self.construct_solutions()
            self.update_pheromones(solutions)
            for solution in solutions:
                value, weight = self.evaluate_solution(solution)

                values_list.append(value)
                weights_list.append(weight)

                if weight <= self.max_weight:
                    refined_solution, refined_value, refined_weight = self.local_search(solution)

                    if refined_value > current_best_value:
                        current_best_value = refined_value
                        current_best_solution = refined_solution

        self.draw_weight_to_values(weights_list, values_list)
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
        self.pheromone *= (1 - self.decay)
        for solution in solutions:
            value, weight = self.evaluate_solution(solution)
            if weight <= self.max_weight:
                for i, included in enumerate(solution):
                    if included:
                        self.pheromone[i] += value / weight

    def evaluate_solution(self, solution):
        value = np.dot(self.values, solution)
        weight = np.dot(self.weights, solution)
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


def tuning_process_grid_search(num_items, values, weights, max_weight):

    alphas = [0.5, 1, 2, 3, 4, 5]
    betas = [1, 2, 3, 4, 5]
    decays = [0.1, 0.2, 0.3, 0.4, 0.5]
    num_ants = [num_items, num_items*2, num_items*3, num_items*4, num_items*5, num_items*6, num_items*7, num_items*8, num_items*9, num_items*10]
    num_iterations = [100]

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

    number_of_runs = 1

    #optimal 309
    aco_knapsack_p01 = ACO_Knapsack(num_items=num_items_P01, values=values_P01, weights=weights_P01,
                                    max_weight=max_weight_P01, num_ants=num_items_P01*2, num_iterations=100, decay=0.1,
                                    alpha=0.5,
                                    beta=2, optimal_value=309)

    run_and_print_result(aco_knapsack_p01, number_of_runs)
    #tunning_process_grid_search(num_items_P01, values_P01, weights_P01, max_weight_P01)

    #optimal 51
    aco_knapsack_p02 = ACO_Knapsack(num_items=num_items_P02, values=values_P02, weights=weights_P02,
                                    max_weight=max_weight_P02, num_ants=num_items_P02*2, num_iterations=100, decay=0.3, alpha=0.5,
                                    beta=2, optimal_value=51)

    #run_and_print_result(aco_knapsack_p02, number_of_runs)
    #tunning_process_grid_search(num_items_P02, values_P02, weights_P02, max_weight_P02)



    #optimal 1735
    #Alpha: 0.5, Beta: 1, Decay: 0.1, Ants: 14, Iterations: 100, Value: 1735
    aco_knapsack_p06 = ACO_Knapsack(num_items=num_items_P06, values=values_P06, weights=weights_P06,
                                    max_weight=max_weight_P06, num_ants=num_items_P06*2, num_iterations=100, decay=0.1, alpha=0.5,
                                    beta=1,optimal_value=1735)

    #run_and_print_result(aco_knapsack_p06, number_of_runs)
    #tunning_process_grid_search(num_items_P06, values_P06, weights_P06, max_weight_P06)


    #optimal 1458
    #Alpha: 1, Beta: 3, Decay: 0.4, Ants: 30, Iterations: 100, Value: 1458

    # aco_knapsack_p07 = ACO_Knapsack(num_items=num_items_P07, values=values_P07, weights=weights_P07,
    #                                 max_weight=max_weight_P07, num_ants=num_items_P07*2, num_iterations=100, decay=0.4, alpha=1,
    #                                 beta=3, optimal_value=1458)
    #
    # #tunning_process_grid_search(num_items_P07, values_P07, weights_P07, max_weight_P07)
    # run_and_print_result(aco_knapsack_p07, number_of_runs)


    # #Alpha: 1, Beta: 4, Decay: 0.5, Ants: 216, Iterations: 100, Value: 13549094
    # #Alpha: 2, Beta: 5, Decay: 0.3, Ants: 144, Iterations: 100, Value: 13549094
    # Alpha: 3, Beta: 3, Decay: 0.3, Ants: 192, Iterations: 100, Value: 13549094
    # Alpha: 3, Beta: 3, Decay: 0.3, Ants: 216, Iterations: 100, Value: 13549094
    # Alpha: 3, Beta: 3, Decay: 0.3, Ants: 240, Iterations: 100, Value: 13549094
    #optimal 13549094

    aco_knapsack_p08 = ACO_Knapsack(num_items=num_items_P08, values=values_P08, weights=weights_P08,
                                   max_weight=max_weight_P08, num_ants=192, num_iterations=100, decay=0.3, alpha=3,
                                   beta=3,optimal_value= 13549094)

    #run_and_print_result(aco_knapsack_p08, number_of_runs)
    #tunning_process_grid_search(num_items_P08, values_P08, weights_P08, max_weight_P08)


if __name__ == "__main__":
    main()



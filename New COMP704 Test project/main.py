import numpy as np
from numpy import load
import gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import ga
import random
import os.path

# human, rgb_array
render_mode = 'human'
env = gym.make('Asterix-ram-v0', render_mode=render_mode)
useOptimization = False


cur_steps = 0
cur_learn_rate = 0
cur_step_total = 0
cur_discount_factor = 0
generation_count = 0
average_generation_reward = []
average_generation_episodes = []
learning_rates = []
global_average_reward = []
discount_factors = []
steps = []


class BaselineWrap(object):
    def __init__(self, ppo):
        self.ppo = ppo

    # Pick the best action for the AI
    def predict(self, env, obs):
        action, _states = self.ppo.predict(obs, deterministic=True)
        return action, _states


# Take the AI through each step of the environment
def run_environment(env, agent, model):
    global render_mode
    observation = env.reset()
    counter = 0
    cur_reward = 0
    disqualifier = 0

    for step in range(5000):
        action, state = agent(env, observation)
        observation, reward, done, info = env.step(action)
        cur_reward += reward
        # prevent render error/warning from occurring
        if render_mode is not 'human':
            env.render()

        # Look at each action of agent, returns no action (1) for 3 steps then it will be given a disqualifier
        if action == 1:
            counter += 1

        if counter > 3:
            disqualifier += 1

        if done:
            print("ended after {} steps".format(step))
            obs = env.reset()
            return step, disqualifier, cur_reward

    env.close()


def train_and_test(individual, filepath, repeats=10):
    global cur_steps, cur_learn_rate, cur_step_total, cur_discount_factor, render_mode
    cur_disqualifier = 0
    reward = 0
    cur_reward = 0
    monitor_wrapper = Monitor(env)
    # Train the model
    print("Training...")

    # learning_rate=individual[1], 0.01, gamma=individual[2], 0.5
    model = DQN("MlpPolicy", monitor_wrapper, verbose=1, learning_rate=individual[1], gamma=individual[2])

    # callback every quarter of iteration to show progress without being too long
    eval_frequency = individual[0] / 4

    # update variable for scatter graph
    cur_learn_rate = individual[1]
    cur_discount_factor = individual[2]
    cur_step_total = individual[0]

    # check if training data exists before creating a new file and overriding data
    if os.path.exists(filepath):
        print('load data')
        model = DQN.load('TrainingData', env=env)

    # Stop training if the AI reaches a constant or highest reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=650, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=round(eval_frequency), deterministic=True, render=False)

    # stop environment from changing during mid testing
    if render_mode is not 'human':
        model.learn(total_timesteps=individual[0], log_interval=round(eval_frequency), callback=eval_callback)


    # check cur step is updatding to save data later
    print('cur_steps: ', cur_steps)
    agent = BaselineWrap(model)

    scores = []
    # the returned values are stored in different valuesto prevent them from being added together
    for epsiode in range(repeats):
        score, disqualifier, reward = run_environment(env, agent.predict, model)

        scores.append(score)

        if cur_disqualifier == 0:
            cur_disqualifier = disqualifier

        # save data during its best performance and make sure it doesn't override itself during testing
        if reward is not None and reward >= cur_reward and disqualifier <= cur_disqualifier and render_mode is not 'human':
            cur_disqualifier = disqualifier
            print('saved data')
            cur_reward = reward
            print('cur_reward: ', cur_reward)
            model.save('TrainingData')

    convert_data_to_csv()

    # scores or score may not be returned
    if scores is None:
        return 0
    elif len(scores) is 0:
        return 0
    elif score is None:
        return 0
    else:
        return sum(scores) / len(scores), disqualifier


def my_select(population, cache):
    # this is an example of how you could use the cache/calculate fitness
    individual = []
    path = "D:/COMP704 Atari AI/COMP_704_Atari_AI/New COMP704 Test project/TrainingData.zip"
    p1 = random.choice(population)
    p2 = random.choice(population)
    p1_fitness, p1_discounter = train_and_test(p1, path, 5)
    p2_fitness, p2_discounter = train_and_test(p2, path, 5)

    print("population 1 discounter: ", p1_discounter)
    print("population 2 discounter: ", p2_discounter)

    # tournament, comparing population
    if p1_fitness > p2_fitness and p1_discounter < p2_discounter:
        individual = p1
        print("p1 ", individual)
    elif p2_fitness > p1_fitness and p2_discounter < p1_discounter:
        individual = p2
        print("p2 ", individual)
    else:
        individual = random.choice([p1, p2])
        print("random ", individual)

    # can't use list as key for a dictonary
    if len(cache) > 0 and individual not in cache:
        cache[str(individual)] = train_and_test(population[individual])

    # we need to return what the algorithm selected
    return individual


def access_table(filepath, data, canread):
    df = data
    # always update the csv to ensure data matches graph
    df.to_csv('data.csv')

    if canread is True:
        print("reading csv")
        df = pd.read_csv(filepath)
    elif canread is False:
        print("create csv file")
        df.to_csv('data.csv')

    # filtering data for graph
    filter_csv_data(len(df.columns), df)


def convert_data_to_csv():
    data = load('./logs/evaluations.npz')
    df = pd.DataFrame.from_dict({item: data[item] for item in data.files}, orient='index')

    lst = data.files
    for item in lst:
        print(item)
        print(data[item])

    # Access csv table in project for filtering
    access_table('D:/COMP704 Atari AI/COMP_704_Atari_AI/New COMP704 Test project/data.csv', df, False)


def filter_csv_data(column_amount, data):
    average_episodes = []
    average_rewards = []
    row = 0
    global average_generation_reward, average_generation_episodes, global_average_reward \
        , learning_rates, cur_learn_rate, cur_step_total, cur_discount_factor, discount_factors, steps

    print(column_amount)

    for i in range((column_amount + 3)):
        # ensure for loop is past all columns to send plot average graph
        if i == column_amount + 2:

            # plot line graph of AI's performance
            plot_graph(average_episodes, average_rewards, "average steps", "average reward",
                       'Feedback calls within iteration', "average of steps and rewards after 3000 episodes")

        elif i < column_amount + 1:
            y1_array = []
            y2_array = []

            if row < column_amount:
                print("filter " + str(data.iloc[0][i]))
                # remove brackets from string so that each number can be converted to float
                row_1 = str(data.iloc[1][row]).replace('[', '').replace(']', '')
                row_2 = str(data.iloc[2][row]).replace('[', '').replace(']', '')

                # move every number from string to separate element in array
                new_y1_array = row_1.split()
                new_y2_array = row_2.split()

                # filter elements on row 1 to ensure that they are numbers
                for y1 in new_y1_array:
                    if y1 != "[" and y1 != "]" and y1 != " " and y1 != ".":
                        print(y1)
                        y1_array.append(float(y1))

                # filter elements on row 2 to ensure that they are numbers
                for y2 in new_y2_array:
                    if y2 != "[" and y2 != "]" and y2 != " " and y2 != ".":
                        print(y2)
                        y2_array.append(float(y2))

                # calculate average for graph to use
                average_rewards.append(sum(y1_array) / len(y1_array))
                average_episodes.append(sum(y2_array) / len(y2_array))

                # collecting data for scatter graph reward to learn rate
                global_average_reward.append(sum(y1_array) / len(y1_array))
                learning_rates.append(cur_learn_rate)
                discount_factors.append(cur_discount_factor)
                steps.append(cur_step_total)

                row += 1

    # average of reward and episode over iteration
    average_generation_reward.append(sum(average_rewards) / len(average_rewards))
    average_generation_episodes.append(sum(average_episodes) / len(average_episodes))


def plot_graph(row1, row2, row1name, row2name, xlabel, title):
    print("Show graph")
    plt.figure(num=0, dpi=120)

    plt.plot(range(len(row1)), np.array(row1), label=row1name)
    plt.plot(range(len(row2)), np.array(row2), label=row2name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('results')
    plt.legend()
    plt.show()


def plot_scatter_graph(xlabel, ylabel, title, xaxis, yaxis, colour):
    print("Show scatter graph")
    fig, ax = plt.subplots()
    ax.scatter(xaxis, yaxis, color=colour)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


def create_csv_file(learn_rate, generation_reward, step_total, discount_factor):
    df = pd.DataFrame.from_dict({'learn rates': learn_rate, 'step total': step_total, 'discount factor': discount_factor
                                    , 'reward': generation_reward}, orient='index')

    df.to_csv('parameter_data.csv')
    return df


def clear_lists():
    global average_generation_episodes, average_generation_reward, learning_rates, global_average_reward, \
        discount_factors, steps

    average_generation_episodes.clear()
    average_generation_reward.clear()
    learning_rates.clear()
    global_average_reward.clear()
    discount_factors.clear()
    steps.clear()


if __name__ == "__main__":

    print("evaluating...")

    # Train with the list of fixed parameters for set number of episodes, [step total, learn rate, discount factor]
    if useOptimization is not True:
        train_and_test([10000, 0.5, 0.752],
                       "D:/COMP704 Atari AI/COMP_704_Atari_AI/New COMP704 Test project/TrainingData.zip", 3000)

        # plot graphs to show the AI progression after the training
        plot_scatter_graph("learn rate", "average reward", "average reward over learn rate",
                           learning_rates, global_average_reward, 'r')
        plot_scatter_graph("discount factor", "average reward", "average reward over discount factor"
                            , discount_factors, global_average_reward, 'b')
        plot_scatter_graph("step total", "average reward", "average reward over step total"
                           , steps, global_average_reward, 'g')

        # ensure that their is no data contamination when plotting graphs
        clear_lists()

    elif useOptimization is True:
        # loop through each generation to improve AI performance
        # 12 generations with a population of 4 individuals
        generations = range(12)
        population = ga.generate_population(4, ga.example_generator)

        for generation in generations:
            generation_count += 1
            population, fitnesses = ga.next_population(population, my_select)
            print(population, fitnesses)

            plot_graph(average_generation_episodes, average_generation_reward, "average episode", "average reward",
                       'iterations in generation', "average reward and steps per generation " + str(generation_count))

            # store data in a csv file
            data = create_csv_file(learning_rates, global_average_reward, steps, discount_factors)
            print("learn rate: ", data.iloc[0][0:])
            print("average reward: ", data.iloc[3][0:])
            print("steps: ", data.iloc[1][0:])
            print("discount factor: ", data.iloc[2][0:])

            # plot graphs to show the AI progression through the generation
            plot_scatter_graph("learn rate", "average reward", "average reward over learn rate generation: "
                               + str(generation_count), learning_rates, global_average_reward, 'r')
            plot_scatter_graph("discount factor", "average reward", "average reward over discount factor generation: "
                               + str(generation_count), discount_factors, global_average_reward, 'b')
            plot_scatter_graph("step total", "average reward", "average reward over step total, generation: "
                               + str(generation_count), steps, global_average_reward, 'g')

            # ensure that their is no data contamination when plotting graphs
            clear_lists()

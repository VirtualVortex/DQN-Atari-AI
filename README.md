# DQN AI playing Asterix Atari game
This repo contains my COMP704 Machine Learning AI project, which is a DQN AI, learning how to play the Atari game Asterix. 

**Requirements:**

When creating a new project for this agent ensure that the version of is python 3.7 64 bit as 32 bit will prevent modules such as gym[atari] from being installed.

Matplolib - for plotting graphs of AI's progress.

ALE-py - for allowing python to emulate Atari game/systems.

stable_baselines3 - for Deep Q Network calulations.

gym - for providing reinforcement environments that the agent can interacte with.

gym[atari] - this library functions similar to the gym library however, it provides emulated environments of Atari 2600 games.

numpy - for advanced maths functonality, for exmaple, libraries like stable_baselines3 need them to preform their maths heavy methods.

numpy import load - used to convert one file type to another type. For example, converting npz to csv.

os.path - for file navigation and checking if file exists.

pandas - for creating, reading and writing of CSV files.

random - for generating random values.

ga - custome genetic algorithm calls for genetic algorithm optimization, needed for AI to function and is a part of the repo.

EvalCallback - used to make evaluations during trainig so that users can save and store data.

StopTrainingOnRewardThreshold - used to tell the DQN AI when to stop training when it reaches a certain reward value.

stable_baselines3.common.callbacks - both EvalCallback and StopTrainingOnRewardThreshold are from this part of the stable_baselines3 library.

stable_baselines3.common.monitor import Monitor - used to ensure that the environment is kept in a monior wrapper for correct use of training data.

\
**How to use the AI:**

- in ga.py on line 17 to 19 - the user can adjust the parameters for the example generation that the AI will use.

- in main.py on line 14 - the user can set the render mode, during use which will affect how the AI uses and saves data.

- in main.py on line 15 - the user can set the environment that the AI will use.

- in main.py on line 16 - the user can tell the AI to use parameter optimization testing (using Genetic Alogorithms).

- in main.py on line 78 - the user can tell the AI what wrapper to use for the environment, to ensure data is used properly.

- in main.py on line 83 - contains the DQN AI function as well as its parameters that can be adjusted by the system's other methods.

- in main.py on line 96 - the user can set the name of the file containing the training data for the AI's use when training.

- in main.py on line 99 - the user can set the reward threshold to ensure the AI stops training in the current episode if it gets the highest possible reward.

- in main.py on line 100 - the user can use the eval callback function for when to return the AI's training results during the training session. The when to make a eval callback is controlled via code in the training function.

- in main.py on line 128 - the user can set the name of the file that will contain the saved training data.

- in main.py from line 312 to 313 - Set the AI's fixed parameters for AI in the first parameter, in the second one set the file path of the training data and in the third parameter set how many episodes you want the AI to perform.

**Disclaimer** - The code should already contain default settings and training data so you can simply press run and the AI should work.

\
**Generate csv files:**

- in main.py on line 177 and 146 - the user can set the name of the csv that will contain the AI's output values.

- in main.py on line 191 - the user can set the file path for which npz file (containg the AI's results) should be converted to csv.

- in main.py on line 200 - set the first parameter to the file path of the csv file for the graph to access the csv file to save a new one or update it. On the second paragraph is the data that the panda load function returns and the third parameter can be set to true to tell it to read the existing file or false to read the newly created data.



\
**Plot graphs:**

- in main.py on line 263 - contains the function for plotting line graphs, this function accepts string and list parameters for labels and row data.

- in main.py on line 276 - contains the function for plotting scatter graphs, this function accepts string and list parameters for labels and row data. 

- in main.py from line 316 to 321 - you can set the scatter graph parameters with out optimisation that will display the title, lables and the data for the x and y axis.

- in main.py from line 348 to 353 - you can set the scatter graph parameters with optimisation that will display the title, lables and the data for the x and y axis.

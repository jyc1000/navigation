# Navigation using Deep Q-Learning

## Project Details
This project trains an agent to navigate the world to collect as many yellow bananas as possible.

### State space:
- 37 dimensions 
  - One dimension for velocity and the rest for ray based perception of the world infront of the agent.

### Action space:
- 4 dimensions (discrete)
  - 0: move forward
  - 1: move backward
  - 2: turn left
  - 3: turn right

### Rewards:
- +1 for collecting a yellow banana
- -1 for collecting a blue banana

## Getting Started:
Below are the necessary python version and python libraries.
  - python: 3.6
  - tensorflow
  - Pillow
  - matplotlib
  - numpy
  - jupyter
  - pytest
  - docopt
  - pyyaml
  - protobuf: 3.5.2
  - grpcio: 1.11.0
  - pandas
  - scipy
  - ipykernel
  - unityagents: 0.4.0
  - torch: 1.10.2

## Instructions
  ### How to run the code:
  1. In Navigation.py, set the directory to the Banana environment file in the code shown below  
  env = UnityEnvironment(file_name="Banana.app")
  using appropriate files for the operating system.
  2. In Navigation.py, set test_mode variable to False for training. Set test_mode variable to True to load pre-trained model and test its performance in the environment.
  3. Finally run Navigation.py




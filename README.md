<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project">
    <img src="images/logo.png" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">Quadruped Robot Project :smile:
</h3>

  <p align="center">
    Exam project of CPSP by 
     Andrea Manfroni,
     Fabrio Grimandi, 
     Giovanni Oltrecolli, 
     Stefano Maggioreni.
    <br />
    <!--<a href="https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project"><strong>Explore the docs »</strong></a> -->
    <br />
    <!--<a href="https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project">View Demo</a> -->
    -
    <a href="https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    -
    <a href="https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
  ## Table of Contents
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#overview-of-Key-Files">Overview of Key Files</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>



<!-- ABOUT THE PROJECT -->
## About The Project
Starting from the URDF model, the goal is to design and train a neural network to enable the quadruped robot to move forward in a straight line without encountering obstacles. Once this initial phase is complete, the plan is to proceed with the Sim-to-Real transfer. This involves deploying the trained model onto a Raspberry Pi 4 embedded within the 3D-printed quadruped, allowing the movements achieved in simulation to be replicated in the real world.

<details>
  <summary><b>Isaac Gym Overview</b></summary>
  <ol>
    Isaac Gym is NVIDIA’s prototype physics simulation environment for reinforcement learning research. It allows developers to experiment with end-to-end      GPU accelerated RL for physically based systems. Unlike other similar ‘gym’ style systems, in Isaac Gym, simulation can run on the GPU, storing results in GPU     tensors rather than copying them back to CPU memory. Isaac Gym includes a basic PPO implementation and a straightforward RL task system that can be used with      it, but users may substitute alternative task systems and RL algorithms. 

  **More information on** 
    https://developer.nvidia.com/isaac-gym
</ol>
</details>


<details>
  <summary><b>SKRL Overview</b></summary>
  <ol>
    Skrl is an open-source library for Reinforcement Learning written in Python. It allows loading and configuring NVIDIA Isaac Gym environments, enabling agents’ simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run.
    
  <b>Please, visit the documentation for usage details and examples:</b> <a href="https://skrl.readthedocs.io">https://skrl.readthedocs.io</a>
  </ol>
</details>


<p align="right"><a href="#readme-top">back to top</a></p>



<!--### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- GETTING STARTED -->
## Getting Started

Instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites
* OS: Ubuntu 20.04

* Python Version: 3.8.10

* Pytorch Version: 2.2.1

* Isaac Gym: [guide for the installation](https://MrDerrick-007.github.io/Quadruped_Robot_CPSP_Project/install.html) 

* Skrl: [installation guide](https://skrl.readthedocs.io/en/latest/intro/installation.html)

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project.git
   ```
2. Go inside "isaccgymenvs" folder
   ```sh
   cd Quadruped_Robot_CPSP_Project/isaacgym/isaacgym_envs/isaacgymenvs
   ```
3. Run the simulation with
   ```js
   python3 torch_anymal_ppo.py 
   ```
<p align="right"><a href="#readme-top">back to top</a></p>

### Checkpoint
You can choose whether to load a pre-trained model from a provided checkpoint or to start a simulation from scratch.
To do this, you need to modify the following line of code in the `anymal.py` file.
 ```python
    agent.load("./runs/torch/Anymal/buono/checkpoints/agent_82800.pt")
   ```
More specifically, if you don't want to use any checkpoint, simply comment out the following line of code. On the other hand, if you want to use a different checkpoint, you just need to modify the path to the selected checkpoint.

**Video demonstration with the checkpoint simulation**
![](videos/final-training.gif)

****Video demonstration without checkpoint****
![](videos/no-trainig1.gif)

<!-- USAGE EXAMPLES -->
## Overview of Key Files

In this section, you can see an overview of the most important file in Isaac Gym folder.

### **Anymal.py:**
In the file "anymal.py," the main functions for initializing and training the environment are listed: `create_sim`, `pre_physics_step`, and `post_physics_step`. These are generic functions present in any environment training setup.

The first function, `create_sim`, is used for initialization. It is responsible for creating the terrain (in our case, a flat, obstacle-free surface) and the environment, with the URDF correctly modified and implemented.

During training, the other two functions are continuously modified. The first, `pre_physics_step`, applies the calculated actions, i.e., it assigns the desired positions to the Degrees of Freedom (DoF). The second, `post_physics_step`, involves three additional generic functions for every training session: `reset_idx`, `compute_observation`, and `compute_reward`.

The function `reset_idx` is applied only when the environment needs to be reset to its initial condition, due to various factors that will be explained later. `compute_observation` reloads the current state value onto the tensor. This function is generic for all environments and is specialized via the `compute_anymal_observation` function. `compute_reward` calculates the rewards, which will be explained later.

<details>
<summary><b>More info</b></summary>
  
**Introduction to Rewards:**

The reward calculations and reset conditions are specifically handled in the `compute_anymal_reward` function. Rewards consist of the calculation of different parameters, each with its own purpose, which are used by the environment for proper training, aiming to achieve the highest possible score. Rewards are divided into positive rewards and penalties. The first type includes positive values that indicate to the environment when movements are performed correctly, while penalties are strictly negative values used to decrease the reward when incorrect movements, as defined in the code, are performed.

---

**List with explanation of all rewards:**

- **Velocity Tracking Reward:** First, the current movement velocity of the environment along the x and y axes (with the z axis perpendicular to the ground) is calculated. Then, the difference between the actual velocity and the desired velocity, set during the creation of the environment, is determined. In the `anymal.yaml` file, the desired velocity ranges along the x and y axes can be set under "randomCommandVelocityRanges." Since this is a reward, the value is made positive by squaring it and multiplying it by its parameter, training the environment to approach the desired velocity as closely as possible.

- **Orientation Penalty:** This penalty calculates the torso's orientation, training the environment to keep the torso as parallel to the ground as possible. The coefficient multiplied by this parameter is negative. In an ideal upright position, the robot's local z-axis is aligned with the global gravity direction. This means the x and y components of gravity in the robot's local reference frame should be close to zero to maximize the reward.

- **Torque Penalty:** This penalty trains the environment to make movements as smooth as possible by using less energy and minimizing movement of the main torso.

- **Joint Acceleration Penalty:** This penalty is related to the acceleration of the robot's DoFs. It encourages the agent to minimize sudden velocity changes in the joints by controlling the difference between the last two requested velocities, contributing to smoother and more controlled movements while minimizing vibrations.

- **Stumbling Penalty:** This penalty tracks tripping or contact errors by the robot's "legs" during movement. It is designed to identify situations where the robot's feet (or other designated contact points) interact with the ground abnormally, such as slipping sideways or making unwanted forceful contacts.

- **Action Rate Penalty:** This penalty operates similarly to joint acceleration. Instead of calculating the difference between the last velocities, it calculates the difference between the DoF positions. This encourages the environment to perform smooth movements with a smaller amplitude, avoiding abrupt joint movements.

- **Cosmetic Penalty:** This penalty forces the environment to keep the shoulder movement of the robot as close as possible to the initial amplitude, leading to a natural and smooth motion.

- **Air Time Reward:** This reward is important for training the environment to keep all four feet in the air for a certain time range. A penalty is applied if any foot stays in the air too long or too briefly. This improves the fluidity and reduces awkwardness in the environment's movement.

- **Symmetric Penalty:** This penalty improves the symmetry of the environment's movement. It uses the air time of the feet, comparing the front left foot with the rear right foot and the front right foot with the rear left foot. The greater the difference in air time between the feet, the higher the penalty value. This forces the environment to make the feet touch the ground at the same time, preventing it from finding any walking pattern different from the desired, symmetric one.

---

To achieve the most desired movement, an important factor is the weight of the rewards, which are the coefficients multiplied by the parameters described above. Since all rewards and penalties are eventually summed into a single value, to avoid unwanted movements, none of the rewards should dominate the others. For example, if the "action rate penalty" gives too high a value compared to "joint acceleration," the environment will tend to focus only on the former and neglect the latter. This would result in small movements with sudden velocity changes.

To understand the weight of all parameters, we used the "wandb" library to save and plot all the parameters, monitoring the magnitude of individual rewards and the total reward simultaneously.

Therefore, multiple simulations must be run while continuously varying the reward weights until a satisfactory result is achieved. The weights are specified in the `anymal.yaml` file or directly in the `anymal.py` file.

---

**List of reset conditions:**

- **Time Out:** This parameter tracks the steps of each environment. If the environment reaches the maximum step count without a reset due to other factors, the reset condition is triggered, returning the system to its initial state to start a new episode. This limits the duration of episodes and allows the learning agent to update its strategy with defined-length episodes.

- **Too Low in the 'Z' Axis:** This parameter forces a reset when the main body gets too close to the ground. It ensures the environment keeps the main body raised a minimum distance from the ground, aiming for a satisfactory movement.

- **Pitch and Roll:** Similar to the previous reset condition, this parameter calculates the pitch and roll of the environment. A reset occurs if the body tilts excessively forward or sideways. This reset condition was chosen because the available motors lack sufficient power, preventing training in situations where the environment needs to rise from a fall. By continuously resetting, we ensure the environment remains as parallel as possible to the ground throughout the movement, preventing overstraining the motors.
</details>

---

### Anymal.yaml

This YAML configuration file is used to define the parameters and settings for the simulation of the robot in an environment, as part of a reinforcement learning setup in Isaac Gym.

<details>
<summary><b>More info</b></summary>
### General Structure:
1. **name**: Specifies the name of the object to be simulated, which is "Anymal" (a robot model).
   
2. **physics_engine**: Refers to the physics engine to be used, with its configuration coming from an external file (`config.yaml`).

3. **env**: This section defines the environment settings.
   - **numEnvs**: The number of environments that will be trained simultaneously (default is 128 if not specified).
   - **envSpacing**: The spacing between each environment in the simulation.
   - **clipObservations**: Limits the observation values for training.
   - **clipActions**: Limits the action values for training.
   - **plane**: Physical properties of the ground, including friction and restitution (bounce).
   - **baseInitState**: The initial state of the robot in the environment, including its position (x, y, z), rotation (as a quaternion), and velocities (linear and angular).
   - **randomCommandVelocityRanges**: Defines the range of velocities the robot will be able to achieve in the x and y directions (used in training).
   - **control**: Controls for robot movement, including PD controller parameters for stiffness and damping, as well as the control frequency.
   - **defaultJointAngles**: Specifies the default target joint angles when no action is taken (i.e., action = 0).
   - **urdfAsset**: Configuration related to the URDF (Unified Robot Description Format) model, such as whether to collapse fixed joints or fix the base link.
   
4. **learn**: Contains learning-specific parameters.
   - **rewards**: Defines reward scaling factors for linear and angular velocity, and torque, influencing how rewards are calculated during training.
   - **normalization**: Scales different physical quantities like linear and angular velocity to adjust the agent's learning dynamics.
   - **episodeLength_s**: Specifies the length of each episode (in seconds).
   
5. **viewer**: Settings for the camera view during simulation.
   - **refEnv**: The reference environment for the camera view.
   - **pos**: The camera position.
   - **lookat**: The point the camera will focus on.

6. **enableCameraSensors**: Indicates whether camera sensors are enabled in the environment.

7. **sim**: Simulation parameters.
   - **dt**: Time step of the simulation.
   - **substeps**: The number of substeps per simulation update.
   - **up_axis**: Defines the up-axis (in this case, "z").
   - **use_gpu_pipeline**: Whether to use GPU-based simulation for faster processing.
   - **gravity**: Gravity values to apply to the simulation (standard gravity in the z-direction).
   - **physx**: Configuration related to the NVIDIA PhysX simulation, including the number of threads, solver type, and GPU usage.
   
8. **task**: Defines task-specific randomization parameters and settings.
   - **randomize**: Specifies whether to apply randomization.
   - **randomization_params**: Defines how and when to randomize different simulation parameters (e.g., noise in observations and actions, gravity, physical properties of the robot).
     - Randomization is done using Gaussian distribution or scaling methods for various parameters, including gravity, friction, damping, and more.
</details>

---
### torch_anymal_ppo.py

This script is set up to train a reinforcement learning (RL) agent using the Proximal Policy Optimization (PPO) algorithm on the Isaac Gym environment for a robot (likely `Anymal`). The script imports several components from the `skrl` library to define the agent, the environment, the memory buffer, and the RL trainer. Below is a detailed explanation of each section of the code.

<details>
<summary><b>More info</b></summary>

### Key Components:
1. **Dependencies and Libraries**:
   - **`isaacgym`**: This is the library to work with Isaac Gym, NVIDIA’s high-performance physics simulator for training AI agents.
   - **`torch`**: The popular deep learning framework is used for implementing the neural networks that will approximate the agent's policy and value functions.
   - **`wandb`**: Weights & Biases is used for experiment tracking and visualization.
   - **`skrl`**: This is a high-level reinforcement learning library that provides utilities for training agents like PPO, as well as environment wrappers, memory buffers, and schedulers.
   - **`argparse`**: For handling command-line arguments (though it is not utilized in this snippet).

2. **Setting the Random Seed**:
   ```python
   set_seed()  # This ensures that the environment and agent training are reproducible.
   ```

3. **Shared Model (Policy and Value)**:
   The `Shared` class inherits from `GaussianMixin` and `DeterministicMixin`, which allow it to handle both stochastic (Gaussian) and deterministic actions. This model contains:
   - **A neural network architecture** with three fully connected layers (256, 128, and 64 units) with `ELU` activation functions.
   - **Mean layer (`mean_layer`)**: This produces the mean for the action distribution.
   - **Log Standard Deviation (`log_std_parameter`)**: This represents the standard deviation of the action distribution, initialized to zeros.
   - **Value layer (`value_layer`)**: This produces the value estimate for the current state.
   
   The `act` method handles action generation (either from a stochastic Gaussian policy or a deterministic one depending on the role).

4. **Environment Setup**:
   ```python
   env = load_isaacgym_env_preview4(task_name="Anymal")
   env = wrap_env(env)
   device = env.device
   ```
   - The `load_isaacgym_env_preview4` function loads the Isaac Gym environment for the task `Anymal`, and `wrap_env` wraps it for easier use within the RL pipeline.
   - The environment runs on the same device (`device`) as the model, which is determined by Isaac Gym.

5. **Memory Buffer**:
   ```python
   memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
   ```
   - The `RandomMemory` class stores the experiences (observations, actions, rewards, etc.) collected from each environment during training. The memory size is set to 24, and it's designed to handle multiple parallel environments.

6. **Model Initialization**:
   ```python
   models["policy"] = Shared(env.observation_space, env.action_space, device)
   models["value"] = models["policy"]
   ```
   - The model for both the policy and value functions is the same in this case (the `Shared` model). Both the `policy` and `value` use the same neural network architecture.

7. **PPO Configuration**:
   ```python
   cfg = PPO_DEFAULT_CONFIG.copy()
   cfg["rollouts"] = 24  # memory_size
   cfg["learning_epochs"] = 5
   cfg["mini_batches"] = 3
   cfg["discount_factor"] = 0.99
   cfg["learning_rate"] = 3e-4
   cfg["learning_rate_scheduler"] = KLAdaptiveRL
   cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
   cfg["timesteps"] = 24000000  # This is the total number of timesteps for training
   ```
   - **PPO Default Configuration**: The PPO algorithm is configured with standard values like:
     - `learning_rate`: 0.0003 (the learning rate for training the models).
     - `rollouts`: Number of environments to sample from (24 environments in this case).
     - `mini_batches`: Number of mini-batches to use for each update.
     - `discount_factor`: The discount factor (`gamma`), set to 0.99, which determines how much future rewards are discounted.
     - `learning_rate_scheduler`: Uses `KLAdaptiveRL`, which adjusts the learning rate based on the Kullback-Leibler (KL) divergence between the old and new policies.
     - `kl_threshold`: Threshold for the KL divergence, used in adaptive learning rate scheduling.
   
   - **State Preprocessors**: Both the state and value are normalized using `RunningStandardScaler` to help with training stability.

8. **PPO Agent Initialization**:
   ```python
   agent = PPO(models=models,
               memory=memory,
               cfg=cfg,
               observation_space=env.observation_space,
               action_space=env.action_space,
               device=device)
   ```
   - The `PPO` agent is initialized with the specified configuration (`cfg`), models (policy and value), and memory buffer.

9. **Loading Checkpoints**:
   ```python
   agent.load("./runs/torch/Anymal/buono/checkpoints/agent_82800.pt")
   ```
   - This loads a previously saved checkpoint so the agent can resume training from where it left off, rather than training from scratch.

10. **Trainer Configuration**:
    ```python
    cfg_trainer = {"timesteps": 24000000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    ```
    - **Trainer Configuration**: Configures the trainer to run for `24,000,000` timesteps (the number of total timesteps for training) and sets it to `headless` mode (no rendering).
    - The `SequentialTrainer` is used to manage training with the environment and agent.

11. **Training**:
    ```python
    trainer.train()
    ```
    - Starts the training process, where the agent interacts with the environment, collects experiences, and updates its policy using PPO.

---

### Evaluation (commented out):
```python
# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface
# path = download_model_from_huggingface("skrl/IsaacGymEnvs-Anymal-PPO", filename="agent.pt")
# agent.load(path)
# trainer.eval()
```
- If you want to evaluate a pre-trained agent (instead of training), you can download the model from Hugging Face and load it into the agent before running the `eval()` method.
</details>

---
<p align="right"><a href="#readme-top">back to top</a></p>



<!-- ROADMAP -->
## Roadmap
- [x] Create a Ubuntu partition
- [x] Isaac Gym
    - [x] Install
    - [x] Try the first examples
- [x] Analyze Isaac's docs and test the Anymal example
- [x] Create a new environment for the robot
- [x] Import the fixed URDF in the Isaac environment
- [x] Training
    - [x] with Anymal's rewards
    - [x] with custom rewards
- [ ] Sim2Real
    - [x] Build the robot
    - [ ] NN deployment in the Raspberry platform 

See the [open issues](https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project/issues) for a full list of proposed features (and known issues).

<p align="right"><a href="#readme-top">back to top</a></p>



<!-- CONTRIBUTING -->
## Contributing
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MrDerrick-007/Quadruped_Robot_CPSP_Project" alt="contrib.rocks image" />
</a>

<p align="right"><a href="#readme-top">back to top</a></p>



<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right"><a href="#readme-top">back to top</a></p>



<!-- CONTACT -->
## Contact

* Andrea Manfroni - andrea.manfroni@studio.unibo.it
* Fabio Grimandi  - fabio.grimandi@studio.unibo.it
* Giovanni Oltrecolli - giovanni.oltrecolli2@studio.unibo.it
* Stefano Maggioreni - stefano.maggioreni@studio.unibo.it

Project Link: [https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project.git](https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project.git)

<p align="right"><a href="#readme-top">back to top</a></p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

List of resources we find helpful for the project.
* [IsaacGym Github](https://github.com/isaac-sim/IsaacGymEnvs/tree/main)
* [SKRL Guide](https://skrl.readthedocs.io/en/latest/intro/installation.html)
* [IsaacGym GetStart Video](https://www.youtube.com/watch?v=nleDq-oJjGk&t=916s&ab_channel=NVIDIAOmniverseYouTube)
* [Robot Assembly Guide](https://github.com/michaelkubina/SpotMicroESP32)
* [Sebastiano Mengozzi Thesis](https://amslaurea.unibo.it/28648/1/SebastianoMengozzi_Thesis.pdf)
* [Sutton, Barto](http://incompleteideas.net/book/the-book-2nd.html)
* [Laura Graesser - Foundations of Deep Reinforcement Learning](https://www.oreilly.com/library/view/foundations-of-deep/9780135172490/)
<!--* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)
-->
<p align="right"><a href="#readme-top">back to top</a></p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/MrDerrick-007/Quadruped_Robot_CPSP_Project.svg?style=for-the-badge
[contributors-url]: https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project//graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MrDerrick-007/Quadruped_Robot_CPSP_Project.svg?style=for-the-badge
[forks-url]: https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project/network/members
[stars-shield]: https://img.shields.io/github/stars/MrDerrick-007/Quadruped_Robot_CPSP_Project.svg?style=for-the-badge
[stars-url]: https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project/stargazers
[issues-shield]: https://img.shields.io/github/issues/MrDerrick-007/Quadruped_Robot_CPSP_Project.svg?style=for-the-badge
[issues-url]: https://github.comMrDerrick-007/Quadruped_Robot_CPSP_Project/issues
[license-shield]: https://img.shields.io/github/license/MrDerrick-007/Quadruped_Robot_CPSP_Project.svg?style=for-the-badge
[license-url]: https://github.com/MrDerrick-007/Quadruped_Robot_CPSP_Project/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

## Credits
URDF model of [Nicola Russo](https://github.com/nicrusso7)
<p align="right"><a href="#readme-top">back to top</a></p>

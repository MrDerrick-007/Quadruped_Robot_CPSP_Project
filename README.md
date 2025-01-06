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
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Starting from the URDF model, the goal is to design and train a neural network to enable the quadruped robot to move forward in a straight line without encountering obstacles. Once this initial phase is complete, the plan is to proceed with the Sim-to-Real transfer. This involves deploying the trained model onto a Raspberry Pi 4 embedded within the 3D-printed quadruped, allowing the movements achieved in simulation to be replicated in the real world.

<details>
  <summary>Isaac Gym Overview</summary>
  <ol>
    Isaac Gym is NVIDIA’s prototype physics simulation environment for reinforcement learning research. It allows developers to experiment with end-to-end      GPU accelerated RL for physically based systems. Unlike other similar ‘gym’ style systems, in Isaac Gym, simulation can run on the GPU, storing results in GPU     tensors rather than copying them back to CPU memory. Isaac Gym includes a basic PPO implementation and a straightforward RL task system that can be used with      it, but users may substitute alternative task systems and RL algorithms. 
  **More information on** 
    https://developer.nvidia.com/isaac-gym
</ol>
</details>


<details>
  <summary>SKRL Overview</summary>
  <ol>
    Skrl is an open-source library for Reinforcement Learning written in Python. It allows loading and configuring NVIDIA Isaac Gym environments, enabling agents’ simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run.

  **Please, visit the documentation for usage details and examples** https://skrl.readthedocs.io
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
2. Go inside "isaccgym envs" folder
   ```sh
   cd Quadruped_Robot_CPSP_Project/isaacgym/isaacgym envs/
   ```
3. Run the simulation with
   ```js
   python3 torch_anymal_ppo.py 
   ```
<p align="right"><a href="#readme-top">back to top</a></p>



<!-- USAGE EXAMPLES -->
## Usage

**Explanation of the contents of anymal.py:**

In the file "anymal.py," the main functions for initializing and training the environment are listed: `create_sim`, `pre_physics_step`, and `post_physics_step`. These are generic functions present in any environment training setup.

The first function, `create_sim`, is used for initialization. It is responsible for creating the terrain (in our case, a flat, obstacle-free surface) and the environment, with the URDF correctly modified and implemented.

During training, the other two functions are continuously modified. The first, `pre_physics_step`, applies the calculated actions, i.e., it assigns the desired positions to the Degrees of Freedom (DoF). The second, `post_physics_step`, involves three additional generic functions for every training session: `reset_idx`, `compute_observation`, and `compute_reward`.

The function `reset_idx` is applied only when the environment needs to be reset to its initial condition, due to various factors that will be explained later. `compute_observation` reloads the current state value onto the tensor. This function is generic for all environments and is specialized via the `compute_anymal_observation` function. `compute_reward` calculates the rewards, which will be explained later.

---

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



Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

![](videos/final-training.gif)
![](videos/no-trainig1.gif)
![](videos/prob_fixing_joint.gif)

Nota: l'algoritmo di RL che usate è PPO della classe actor-critic

_For more examples, please refer to the [Documentation](https://example.com)_

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

**Enhancing Autonomous Navigation Robustness in**   
**Unstructured Environments using**   
**Quantum-Seeded Procedural Maze Generation**

A Thesis Manuscript presented to the Faculty of   
College of Informatics and Computing Sciences    
BATANGAS STATE UNIVERSITY   
The National Engineering University  
Batangas City 

# TITLE PAGE

In Partial Fulfillment   
of the Requirements for the Degree   
Bachelor of Science in Computer Science

Antony, Aldrich Ryan V.  
Baricante, Mark Angelo R.  
Mirabel, Kevin Hans Aurick S.

John Richard M. Esguerra, MSCS  
Supervisor

April 2026

# **CHAPTER 1**

**INTRODUCTION**

## **1.1 Background of the Study**

Procedural generation (PG) is a method for creating content using algorithms which define the rules of what is being generated. The application of PG can relate to many fields, such as game development, as a means to automate level design and reduce development costs and effort (Zhang et al., 2022). Another application is in the entertainment and simulation domains, where procedural content generation techniques are increasingly used not only for creating diverse virtual environments but also as a means of increasing generality in machine learning by generating varied training environments that improve agent generalization (Risi & Togelius, 2020). Lastly is to leverage procedural generation in order to benchmark reinforcement learning, not only for benchmarking but also as a means for training artificial intelligence models, since for most cases, the procurement of training data could prove to be costly and time-consuming (Cobbe et al., 2020; Schmedemann et al., 2022). By generating diverse and functional environments, PCG helps ML systems learn robust, transferable policies rather than brittle and overfitted ones. A good example is for car training data. As the trend for self-driving cars has been on the rise, there is a need for training data in order for the car to make decisions without human intervention. However, other than hardware limitations and algorithmic constraints, autonomous vehicles face a number of technical and non-technical challenges in terms of real-time implementation, safety, and reliability (Kaur & Rampersad, 2023), which is why efforts have been made to use PG in simulating road networks. All of these highlight the versatility of procedural generation as a tool in different fields like game development, generating terrain in creative fields, and creating diverse simulations.

The key to diversity in PG is randomness. Traditionally, randomness in applications such as video games, simulations, cryptographic protocols, randomized algorithms, and Monte Carlo techniques relies on pseudo-random number generators (PRNGs), which produce deterministic sequences that imitate true randomness and are not only fast but also efficient for computational use (Bhattacharjee & Das, 2022; Fort, 2015). PRNGs are limited to their deterministic nature, not reaching true unpredictability, potentially reducing diversity in generated content, which may contribute to overfitting, a possibility that this study seeks to evaluate, where the agent memorizes the specific layout or the underlying algorithmic patterns of the training environment rather than learning generalizable navigation strategies (Zhang et al., 2018), and to add the security risk it brings up for cryptographic-related contexts, as it is possible to replicate outputs once the seed is known.

More recently, Quantum Random Number Generators (QRNGs) have emerged, taking advantage of the principles of quantum mechanics, based on fundamentally unpredictable phenomena like superposition and measurement collapse, to generate randomness that is theoretically impossible to predict or replicate (Bhattacharjee & Das, 2022; Jozwiak et al., 2024), showing QRNG’s reliability in cryptographic fields. This study hypothesizes that QRNG may provide a more structurally diverse source of randomness

However, despite the potential that QRNGs offer, research on their use has been limited to cryptography and security. Its potential in PG, which primarily relies on randomness to create unpredictable outcomes, has been ignored. PG continues to rely on PRNG as its default source of randomness in order to create or generate outputs such as maps, terrains, simulations, and mazes. While PRNGs today dominate in domains such as cryptography, simulations, and games, the application of QRNG remains unexplored outside security contexts. 

### **Statement of the Problem**

Deep Reinforcement Learning (DRL) agents frequently suffer from overfitting, where they memorize training environments rather than learning generalized navigation logic. This causes agents to fail when encountering novel layouts in unstructured environments. While existing solutions primarily address this through visual domain randomization (changing textures, colors, or lighting), there is a significant lack of research regarding structural randomization.

Standard Procedural Content Generation (PCG) relies on Pseudo-Random Number Generators (PRNGs), which are inherently deterministic and prone to periodicity. It remains distinctively unclear whether this lack of true entropy restricts the Zero-Shot Generalization capabilities of DRL agents. Consequently, there is a need to determine if introducing true stochasticity via Quantum Random Number Generators (QRNGs) can mitigate structural overfitting and improve navigational robustness.

The study seeks to answer the following questions:

1. How do training environments generated via PRNG differ from QRNG in terms of structural complexity metrics and diversity, as quantified through topological, statistical, and distributional approaches?  
2. How do different sources of randomness in the training environment affect the learning rate of various DRL agents?  
3. How does the DRL agent trained on QRNG-seeded environments perform when navigating unseen maze structures compared to the baseline PRNG model?

## **1.2 Objectives of the Study**

The study aims to address the gap by conducting an exploratory benchmark of QRNG against PRNG in maze generations using the recursive backtracking algorithm. By comparing the structural diversity and unpredictability of the generated mazes, the study explores whether QRNGs provide any substantial and measurable benefits in procedural generation for training DRL agents. Significant differences would indicate a potential for QRNGs' use in procedural generative content, while minimal differences would suggest their advantages in this domain are limited, though other applications or contexts may still benefit from quantum randomness.

### **General Objective**

To evaluate the impact of quantum random number generator (QRNG) seeds compared to pseudo-random number generator (PRNG) seeds when applied to procedural generation in training  and evaluating DRL agents to navigate through the maze.

### **Specific Objectives**

1. To determine how different sources of randomness (PRNGs and QRNGs) influence the structural complexity and diversity of the maze training environments.  
2. To determine the extent to which QRNG-based sources impact the training of autonomous navigation models in terms of success rate and rewards over episodes.  
3. To evaluate the performance of DRL agents trained in QRNG‑seeded versus PRNG‑seeded mazes on unseen environments, in terms of success rate, reward, efficiency, and generalization.

## **1.3 Novelty Claims**

This study presents a novel study on the use of Quantum Random Number Generators (QRNGs) as a source of randomness for Procedural Content Generation (PCG), outside of their usual application in cryptography. Existing research is seen to rely on pseudo-random number generators (PRNGs) and focuses primarily on visual domain randomization in training environments in an attempt to improve generalization in Deep Reinforcement Learning (DRL) agents. This study introduces a controlled seed-level substitution experimental setup, where QRNG-generated entropy replaces PRNG seeds while keeping the maze generation algorithm and training pipeline as a constant. The study further contributes by quantitatively analyzing structural diversity using topological and statistical maze metrics and by evaluating the effects of quantum-seeded environments across multiple DRL architectures (DQN, A2C, and PPO) as a means of benchmarking the randomness itself and its effect, rather than model superiority. With this approach, the research reframes quantum randomness as a potential tool for enhancing structural diversity and generalization in reinforcement learning environments.

## **1.4 Scope and Limitations**

The study focuses on evaluating the impact of Quantum Random Number Generator (QRNG) seeds compared to Pseudo-Random Number Generator (PRNG) seeds when applied to procedural generation for enhancing the robustness of Deep Reinforcement Learning (DRL) agents in autonomous navigation.

The randomness sources for the experimental group involve using the Australian National University’s (ANU) Entropy-as-a-Service (EaaS) to procure true quantum seeds. For the control group, the study will use PRNG seeds generated from Python’s built-in deterministic random module. These seeds will drive a recursive backtracking algorithm to generate 2D maze environments. All implementations will be conducted in Python.

To benchmark the effect of these randomness sources on navigational robustness, the study utilizes three distinct DRL architectures: Deep Q-Network (DQN), Advantage Actor-Critic (A2C), and Proximal Policy Optimization (PPO). It is important to note that these models possess different hyperparameters and learning approaches (e.g., value-based vs. policy-gradient). Consequently, this research does not aim to compare the performance of the models against each other (e.g., determining if PPO is better than DQN). Instead, these models serve as benchmarking tools to evaluate the source of randomness itself. The primary objective is to determine if training on QRNG-seeded environments yields consistently more robust agents compared to their PRNG-trained counterparts, regardless of the specific model architecture used. Furthermore, this study explicitly establishes the Pseudo-Random Number Generator (PRNG) utilizing the Mersenne Twister algorithm as the performance baseline. Improvement, in this context is not measured by the agents achieving an arbitrary high score, but is strictly defined by the comparative difference in navigation success rates and generalization capabilities between the QRNG-trained agents (experimental group) and the PRNG-trained agents (control group) when exposed to identical, unseen testing environments.

The study is delimited to 2D maze structures; 3D environments or other complex terrains are not within the scope. Furthermore, subjective factors such as maze playability, user preference, and game-level design are excluded. The evaluation focuses strictly on structural statistical properties and the quantitative performance metrics of the DRL agents. Finally, the study analyzes only seed-level randomness and does not cover long-form RNG sequences or time-series randomness behavior.

## **1.5 Significance of the Study**

The findings of this study may offer valuable insights for several groups. For game developers who could utilize QRNG in order to create more unpredictable and unique maze structures, improving the replayability and quality of the games being developed, especially for infinitely generating worlds, as then the true limit would be the amount of assets for the game to use. Simulation engineers may benefit from the simulation appearing to be more realistic and randomly generated. It can also reduce the structural bias, which will improve the reliability of simulation outputs. This also relates to AI training and benchmarking as a means for creating even more unpredictable scenarios for models to try and overcome in order to improve the quality of the models. Computer science students, as this study may help or inspire students to expand their knowledge on how quantum randomness can affect the quality of the algorithmic output and evaluating QRNG beyond conventional methodologies. Future researchers may use the results of this study to serve as benchmark data on the differences of seed generation randomness between PRNG and QRNG in terms of procedural maze generation. Lastly, to the field of cryptography, as the insights may prove to be useful to improve the implementation of cryptographic systems. This research aligns with SDG 9: (Industry, Innovation, and Infrastructure), as it investigates the possible application of quantum-based randomness to enhance procedural generation and strengthen the reliability of autonomous systems. The research might help in building robust digital infrastructure and advanced AI technologies for simulations and robotics by improving the generalization and consistency of DRL agents through training in various environments.

## **1.6 Definition of Terms**

1. **Procedural Generation.** a technique used in computing to create data algorithmically rather than manually, creating a variety of maze environments for reinforcement learning agent training.  
2. **Pseudo-Random Number Generators.** Deterministic algorithms that produce sequences resembling randomness, used as the baseline source of seeds for maze generation.  
3. **Quantum Random Number Generators.**  Devices that take advantage of quantum phenomena to produce true randomness, applied in the study to generate seeds for maze environments.  
4. **Recursive Backtracking.** A maze generation algorithm that explores and backtracks through cells, chosen in the study for its simplicity, seed sensitivity, and ability to produce solvable perfect mazes.  
5. **Seed.** An initial value that determines the sequence of random numbers, directly influencing the structure of mazes generated in the experiments.  
6. **Environment.** The maze in which an agent operates, generated procedurally using PRNG or QRNG seeds.  
7. **Unstructured Environments.** Environments lacking predictable geometric patterns, serving as the test scenarios where agents must navigate without relying on regular structures.  
8. **Reinforcement Learning.** A machine learning paradigm where agents learn by interacting with environments and receiving rewards, forming the basis of the study’s training approach.  
9. **LiDAR.** A sensor technology that measures distances using laser light in robotics, referenced and simulated in the study as part of the data that the agent receives.  
10. **Generalization.** The ability of agents to apply learned navigation strategies to unseen environments, serving as a key performance metric in the experiments.

# **CHAPTER 3**

**RESEARCH METHODOLOGY**

## 

## **3.1 Research Design**

The study uses a comparative experimental research design to determine how much PRNGs and QRNGs influence the structural outcomes of procedural maze generation and its effect on the performance and generalization of different RL agents. The study uses the seed sources, PRNG and QRNG, as the independent and primary variable by using an identical maze generation algorithm, which is recursive backtracking. Using controlled tests, the study will evaluate the seeds’ statistical randomness along with the generated mazes’ complexity and variability. To extend this analysis toward navigation performance, reinforcement learning agents, namely Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C), are trained within the generated mazes to evaluate how the randomness source of the environment affects learning dynamics, policy effectiveness, and generalization. The combination of RL training and evaluation guarantees that the research assesses the variability and diversity of procedural structures while also methodically investigating their effects on the capacity of autonomous agents to adapt and generalize

## **Data Collection**

This study will collect quantitative data from two randomness sources, a Pseudo-random Number Generator (PRNG) and Quantum Random Generator (QRNG). Each seed will produce one 20 x 20 maze so that the structural differences between the two random number generators can be compared. 

To ensure the fairness of the randomness comparison, the study will standardize the bit-width of the seeds used for both groups to 64-bit integers. PRNG, which is the control group, will use Python’s built-in random module that utilizes the Mersenne-Twister algorithm. This algorithm is chosen because of its proven 623-dimensional equidistribution and a period of 219937\-1, which eliminates the possibility of cycle repetition. The system will generate standard 64-bit integer seeds, which serve as the controlled pseudo-random baseline.

Moreover, the QRNG, which is the experimental group, will use the Australian National University’s Quantum API. This AQN API measures the vacuum fluctuations to generate true random numbers. Since the API outputs data in uint16 (16-bit) blocks, the study will use a bitwise concatenation strategy to match the 64-bit integer seeds of PRNG. This ensures QRNG seeds will have the same value range and bit density as the PRNG seeds. The API has request limit of 100 calls, with up to 1,024 numbers, which enabled the study to gather 71,800 concatenated QRNG numbers.

In terms of reproducibility, all the raw API responses (raw 16-bit chunks, timestamps, and metadata) will be stored in a *raw\_qrng\_logs.json* file. The final output for both groups will be saved on separate CSV files (prng\_seeds.csv and qrng\_seeds.csv). The separation will keep the exact quantum data retrieval instance preserved for auditability and maintainability of the input dataset.

Before utilizing the collected seeds for the recursive backtracking algorithm, the researchers will perform statistical diagnostics to ensure the quality of randomness for both PRNG and QRNG seeds. This validation is an important step to make sure that the bitwise concatenation strategy did not introduce artificial correlations and met the necessary unpredictability standards.

1. **Shannon Entropy:** This metric will measure the density and unpredictability of the generated seeds. Ideally, a truly random sequence of *n* outcomes should have an entropy close to log2(*n*). The researchers will calculate the Shannon entropy of both PRNG and QRNG seeds. High entropy values will indicate that the seeds contain enough randomness and are free from bias that could affect the maze structure. Shannon entropy has the formula: H(X)=-i=1np(xi)logb(p(xi)), where *p(xi)* is the probability of occurrence of outcome *xi*, *n* is the number of distinct outcomes, and *b \= 2* is the logarithm base, for bits. A higher entropy value suggests greater diversity in seed generation, while lower entropy may indicate structural bias that could propagate into maze complexity.  
2. **Autocorrelation:** This ensures the independence between trials. It measures the correlation between a seed (*St* ) and its delayed version (*St+k*). High autocorrelation indicates the previous seed could predict the next seed, which would invalidate the independence of the generated mazes. It also confirms the concatenated QRNG seeds will have statistical independence just like the PRNG baseline.

This study will use the recursive backtracking algorithm as the main instrument for producing the maze dataset. Simulations will be done on local hardware. Since the outputs of QRNG are non-deterministic, all the generated values will be logged and reused as fixed inputs to ensure reproducibility and fairness in comparison with PRNG. For the seed generation, PRNG will use the random module to generate the required number (from power analysis) of PRNG seeds, while the QRNG seeds will be from ANU-QRNG. Both seed groups will be stored in CSV format separately. Once the seeds are ready, each seed will be fed into the Recursive Backtracking algorithm to generate one 20 x 20 maze.  The output mazes will be directly converted to the needed metrics via a script in python and saved in CSV format. The collected dataset will be the basis for comparing the randomness between PRNG and QRNG on procedural maze generation and will serve as the evidence for evaluating the differences in maze structure and unpredictability or extent of the entropy.

## **Deep Reinforcement Learning Implementation**

This experiment tests whether quantum-seeded randomness improves how well navigation agents hold up outside their training data. To do that, the training pipeline compares three well-established Deep Reinforcement Learning approaches: Deep Q-Network (DQN), Advantage Actor-Critic (A2C), and Proximal Policy Optimization (PPO). Together, they cover value-based learning (DQN), policy optimization (PPO), and a hybrid actor–critic setup (A2C).

1. **Environment Configuration.** The maze environments will be wrapped as a custom Gymnasium interface. Unlike visual grid-based inputs, this study utilizes a sensor-based observation space to simulate realistic robotic perception.  
* **Observation Space (10-Dimensional Vector):** The agent receives a normalized vector of 10 floating-point values:  
  * **LiDAR Readings (8 values):** Eight ray-casts distributed radially (0°, 45°, 90°, …) to detect distance to the nearest wall.  
  * **Target Vector (2 values)**: The relative *(x, y)* direction or polar coordinates pointing toward the goal.  
* **Action Space:** A discrete space of four actions: *A \= {Up, Down, Left, Right}.*   
* **Termination Conditions:** An episode terminates if the agent reaches the goal or exceeds a maximum of 500 steps.  
* **Reward Function:** A sparse reward structure is implemented to minimize reward hacking and prioritize shortest-path seeking:  
  * *Rgoal*  \= \+200.0 (Task Completion)  
  * *Rstep*  \= \-0.01 (Base Step Penalty)  
  * *Rcollision*  \= \-1.0 (Wall Collision Penalty)  
  * R*exploration* \= \+0.1 (Exploration Bonus)  
  * R*productive* \= \+2.0 (Productive Exploration Bonus)  
  * R*timeout*  \= \-10.0 (Timeout Penalty)  
2. **Model Architecture.** Since the input is a 1-dimensional feature vector rather than a 2D image, the agents will utilize a Multi-Layer Perceptron (MLP) policy (MlpPolicy) rather than a CNN.  
* **Input Layer:** The network accepts a 10-float observation vector, representing the 8-ray LiDAR readings and the 2-dimensional relative target coordinates.  
* **Output Layer:** Maps the processed features to the 4 discrete actions.  
3. **Training Hyperparameters.**  To ensure the validity of the comparison, Common Environmental Parameters (Table 2\) will be held constant across all agents to ensure they face identical learning conditions. However, Algorithm-Specific Hyperparameters (Table 3\) are empirically determined. This approach ensures that each algorithm operates within its intended stability region.

**Table 2**

*Common Environmental Parameters (applied to all agents)*

| Parameter | Value | Justification |
| :---- | :---- | :---- |
| Total Timesteps | 1 x 106  | Sufficient for convergence in low-dimensional sensor tasks. |
| Discount Factor (ˠ) | 0.99 | Prioritizes long-term goal reaching over short-term survival. |
| Observation Normalization | True | LiDAR readings normalized to \[0, 1\] range to prevent gradient instability. |
| Policy | MlpPolicy | Enables feature extraction from vector-based observations using a multilayer perceptron. |
| Action Space | Discrete(4) | Standard movement {Up, Down, Left, Right}. |

**Table 3**

*Algorithm-Specific Configurations*

| Parameter | DQN | A2C | PPO |
| :---- | :---- | :---- | :---- |
| Learning Rate | 1 x 10\-3 | 7 x 10\-4 | 3 x 10\-4 |
| Batch Size | 512 | N/A (Rollout-based) | 512 |
| Buffer Size | 500,000 | N/A | N/A |
| n\_steps (Rollout) | N/A | 5 | 2048 |
| Exploration / Entropy | ε-greedy (1.0 to 0.05) | Entropy Coeff: 0.0 | Entropy Coeff: 0.01 |
| Network Architecture | MLP (256, 256\) | MLP (64, 64\) | MLP (256, 256\) |

**4\. Comparative Training Protocol.** The study utilizes a Two-Group Comparative Design. For each algorithm (DQN, A2C, PPO), two distinct instances are initialized, resulting in six total trained agents.

* **Control Group *(AgentPRNG)*:** Trained exclusively on the PRNG Training Set (80% of PRNG mazes). This establishes the baseline performance for standard procedural generation.  
* **Experimental Group *(AgentQRNG)*:**  Trained exclusively on the QRNG Training Set (80% of QRNG mazes). This tests the hypothesis that higher entropy in quantum seeds produces more diverse training scenarios, reducing overfitting.  
* In this comparative protocol, the PRNG-trained models serve as the stationary baseline against which quantum advantage is measured. Consequently, the study focuses on the *delta* or relative performance gap between the two groups. Improvement is quantitatively assessed by determining if the ***AgentQRNG*** yields statistically higher mean rewards and success rates than the ***AgentPRNG*** baseline across the validation sets, isolating the quality of the seed entropy as the sole differentiating factor.  
* **Validation:** Training will occur on local hardware and Google Colab for simultaneous training, with model checkpoints saved every 10,000 steps to monitor the learning curve.  
  **5\. Generalization Testing Strategy.** Post training, the weights of all agents are frozen to prevent further learning. The Generalization Gap is then measured by evaluating the agents on the Test Sets (the remaining 20% of unseen mazes).  
* **Intra-Domain Generalization:** *AgentPRNG* is tested on unseen PRNG mazes and *AgentQRNG* is tested on unseen QRNG mazes to measure standard robustness.  
* **Cross-Domain Generalization:** *AgentQRNG* is tested on unseen PRNG mazes to determine if quantum-trained agents are more robust when facing standard structures, and *AgentPRNG* are tested on unseen QRNG mazes.

## **3.6 Evaluation Metrics**

This outlines how the QRNG and PRNG seeds and generated mazes will be evaluated.

1. **Maze Structural Evaluation.** Each generated maze is converted to two graph representations: a cell graph where each path cell is a node and orthogonal adjacency (N, E, W, S) defines edges, and corridor graph where maximal degree‑2 chains are collapsed into single corridor edges, while junctions and dead ends become nodes. This transformation reduces pixel noise and yields corridor‑level features, providing a higher-level abstraction of maze topology.  
   * **Path length.** The shortest path length from the designated start-to-finish using Dijkstra. The mean, median, IQR, and distribution of both PRNG and QRNG groups will be recorded, and visualized with histograms and boxplots.  
   * **Path tortuosity.** Referring to how much the path winds, path tortuosity translates to maze difficulty (Wilson et al., 2021). This will be recorded using the formula Tortuosity=shortest pathManhattan distance. Path tortuosity is often skewed with outliers, this being very winding mazes. To maintain statistical integrity, the median and IQR will be recorded instead of the mean.  
   * **Dead-end count**. This is the number of nodes with degree-1 in a corridor graph. Degree-1 nodes in the corridor graph will be counted, finding the mean, SD, and median of degree-1 nodes in each maze group will be recorded. The distribution will be visualized with histogram and boxplots.  
   * **Junction proportions**. The proportion of nodes with degree-3 and degree-4 relative to the total number of junction nodes.  For each corridor graph, compute p3=degree-3junctionsand p4=degree-4junctions for recording. Group means will be visualized with histograms, and bootstrap with 95% CI.  
   * **Turns and straightaways**. Degree-2 chains are collapsed into corridor edges after recording counts of straight corridors and counts of turns per corridor. Collapsed corridor edges will be identified and classified as straight (no internal turns) or turning (contains a turn). The study will report the histogram and the mean turns.  
2. **Maze Diversity.** The following will be visualized to show if there is a difference in diversity of mazes among PRNG generated mazes and QRNG generated mazes.  
   * **Expressive Range Analysis (ERA)**. Two metrics will be chosen as axes for ERA, Linearity and Leniency. To ensure proper measurement, the cell graph of the maze will be used to compute both. The convex hull area of the ERA cloud of PRNG and QRNG will be visualized and measured, where a larger area indicates greater expressive range, meaning greater diversity.  
     1. Linearity is the fraction of straight corridor length over the total corridor length. To compute, the sum of lengths of degree-2 nodes with no turns divided by the total number of corridors (degree-2 nodes).  
        Linearity=degree-2 nodes with no turnsall degree-2 nodes  
     2. Leniency is the fraction of negative of the sum of all dead-ends divided by the total number of cells.  
        Leniency=-(Dead-end nodes)all nodes  
     3. Scatter Plot of Linearity vs. Leniency, using different colors and semi-transparent markers for PRNG and QRNG points.  
     4. Convex hull is the area of the smallest convex polygon that contains all the points in a group, and will be computed using Gauss area formula A=12|i=1n(xiyi+1-(yixi+1)|.  
3. **Reinforcement learning agent metrics.** These operational metrics capture difficulty beyond static topology.  
1. **Agent Training Metrics.** Three RL models will train on PRNG-generated mazes. A separate training process will be conducted for QRNG-generated mazes. The following will be recorded during training.  
   * **Training Reward over Episodes**. The cumulative rewards per episode of the agents on QRNG and PRNG training of the agents are recorded and are shown in a time-series graph. Plotting this over time highlights learning, showing whether agents improve steadily and when their performance stabilizes.  
     * **Success Rate % (SR) over Episodes.** This shows the progress of learning and generalization of the agents in both QRNG and PRNG mazes.  
2. **Agent Evaluation Metrics.** The AgentQRNG and AgentPRNG will be evaluated in novel mazes of both sources, and the following metrics recorded.  
   * **Success Rate % (SR).** This reflects the agent’s reliability in solving novel tasks  
     * **Reward.** This summarizes the overall navigation quality, combining positive outcomes with penalties for the agent’s inefficiency or invalid moves.  
       * **Amount of Steps Taken.** The amount of steps measures path efficiency, indicating whether solutions are near‑optimal or unnecessarily long.  
       * **Generalization Gap.** *SRintra \- SRcross*, quantifies the drop in success between familiar and unfamiliar maze distributions, which serves as a key indicator of robustness and resistance to overfitting.

## **3.8 Statistical Validation**

This section covers what to do with the evaluated metrics, testing for statistical and practical significance towards maze structures, maze diversity of each group, and solver behavior. A significance level of  \= 0.05 will be used in the study.

### **One variable tests**

**1\. Testing for Normality.** The mentioned data will be tested for normality before undergoing other tests. Normality testing is done to validate assumptions of parametric tests, and adjusted if necessary. The recorded variables will go through the Shapiro-Wilk test, as research has shown that the efficiency of the Shapiro-Wilk test is sensitive to sample size, with low sample counts generating bias in normality analysis (Souza et al., 2023). If Shapiro-Wilk detects non-normality, non-parametric tests will be considered. The Python library scipy.stats has Shapiro-Wilk as one of its tests, which will be used for automation of tests.

**2\. Comparing means.** The study will use independent samples t-test since the difference of structural quality of PRNG mazes and QRNG mazes will be compared, which are independent from each other. The Welch t-test is recommended as best practice when comparing two independent groups, as it allows for unequal standard deviations between groups and has almost as much statistical power as Student's t-test (West, 2021). Independent samples t-test use the formula:  
t \= (x1 \-  x2)s12n1 \+ s22n2  
where x1and x2 are the means of the two samples, s1 and s2 are the standard deviations, and n1 and n2 are the sizes, where equal variances are assumed. If the calculated pα (which is 0.05), there is a significant difference between the two means.  
	In the case of a non-normal distribution, the statistical tool Mann-Whitney U will be used as a nonparametric test. Mann-Whitney U is used to determine if there is a significant difference between two groups by ranking all data points from two independent groups from lowest to highest, assigning average ranks for ties. As a nonparametric alternative to the t-test, Mann-Whitney's U test compares two independent groups without requiring the assumption of normality, though it does assume exchangeability between the groups (Karch, 2021). Mann-Whitney U uses the formulaUc=ncntnc(nc+1)2-Rt and Ut=ncntnt(nt+1)2-Rc, where nc and nt are the sample sizes for the control and treated groups, and Rc and Rt are the sums of the ranks for each group. If p \< α (which is 0.05), reject H₀ → conclude a statistically significant difference.

### **Multivariate testing**

**1\. Permutational Multivariate Analysis of Variance (PERMANOVA)** will be applied towards the metrics that are found to be non-normal distributions. PERMANOVA is a non-parametric method that operates directly on distance matrices. PERMANOVA partitions the variation in the distance matrix among groups and assesses significance through a number of permutations, thereby providing a robust test of group differences without requiring normalization.

The null hypothesis will be tested with 𝞪 \= 0.05: Ho \= There is no significant difference between corridor length, betweenness centrality, and spectral features between PRNG mazes and QRNG mazes.

After PERMANOVA is performed, the primary test statistic reported will be pseudo-F, which comes from the ratio of between-group to within-group variation in the distance matrix. Significance of the pseudo-F will be assessed through permutation tests (1000). If the null hypothesis is rejected, analysis will be done to identify which specific metrics contributed strongly to the result. This will be done by using pair-wise PERMANOVA tests on pairs of metrics.

**2\. Multivariate Analysis of Variance (MANOVA)** will use the mean vectors of normalized structural maze metrics. The goal of MANOVA is to determine whether the means of the dependent variable differ significantly across groups while considering the interrelationships between the variables. This is the multivariate version of ANOVA. Prior to conducting MANOVA, the provided data will first need to be screened and tested against core assumptions such as independence of observations, absence of univariate and multivariate outliers, linearity, multivariate normality, and homogeneity of covariance matrices.

The hypotheses for MANOVA will be tested with a 𝞪 \= 0.05: Null Hypothesis (H0)  \= The means of the dependent variables which are the structural maze metrics are equal across the two groups (PRNG and QRNG). There is no significant multivariate effect of the RNG type on the combined structural metrics of the mazes.

After the one-way MANOVA is performed, the primary test statistic reported will be Wilk’s Lambda (Λ). If the null hypothesis is rejected, a follow up test will be performed for further analysis in order to pinpoint the source of the difference, this is done by performing separate univariate F-tests for each dependent variable in order to determine which is the specific structural metric that contributed to the overall multivariate effect. The effect size for the significant multivariate test will then be reported using partial eta squared (ɳp2).

## **Null Hypothesis (Ho)**

1. There is no significant difference in the structural complexity and diversity of mazes generated using QRNG seeds in comparison to mazes generated using PRNG seeds.  
2. There is no significant difference in the learning curve (success rate and cumulative rewards) between DRL agents trained in PRNG-generated mazes and DRL agents trained in QRNG-generated mazes.  
3. DRL agents trained on QRNG-seeded mazes do not generalize better on unseen maze structures than agents trained on PRNG-seeded mazes.

## **Alternative Hypothesis (H1)**

1. There is a significant difference in the structural complexity and diversity of mazes generated using QRNG seeds in comparison to mazes generated using PRNG seeds.  
2. There is a significant difference in the learning curve (success rate and cumulative rewards) between DRL agents trained in PRNG-generated mazes and DRL agents trained in QRNG-generated mazes.  
3. DRL agents trained on QRNG-seeded mazes generalize better on unseen maze structures than agents trained on PRNG-seeded mazes.

## **Validity and Reliability**

This study ensures the integrity of the experimental results by applying different validity measures. The first one is the construct validity, where the study utilizes verified quantum sources from the ANU Quantum Random Number server. This ensures that the independent variable has a true non-deterministic quantum entropy rather than a mathematically deterministic simulation of it. Another one is the internal validity, which ensures that there will be no differences in the attributes of the maze generation logic. No parameters will be altered between PRNG and QRNG, so both algorithms will be tested fairly. The only change is the input seed since PRNG is using a mathematical algorithm to generate the seeds, while QRNG gets its seeds from a physical quantum event.

In terms of reliability, a balanced dataset of mazes (PRNG mazes and QRNG mazes) and their features will be utilized to make the results generalizable and minimize the statistical outliers. For procedural reliability, the entire workflow will be automated. From seed acquisition up to maze generation, all the way through to the statistical tests, will be run on a single, automated Python pipeline. In this way, human errors will be minimized, and the entire process will be faster and smoother.

## **Constraints and Trade-offs**

Unlike PRNG where the seeds can be generated by just using a mathematical formula, QRNG requires using a quantum computer to generate non-deterministic randomness. Because of this, external API connectivity from verified and reputable quantum providers like ANU Quantum Random Number Server will be utilized. This ensures that the QRNG seeds will still have the non-deterministic randomness since it is generating seeds that are from a real quantum number generator.
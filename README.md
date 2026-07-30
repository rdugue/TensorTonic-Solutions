# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Ralph Dugue's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/kingphito.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| AdaGrad Optimizer | Implement a vectorized AdaGrad update in NumPy with accumulated squared gradients and adaptive per-parameter learning rates. | https://www.tensortonic.com/problems/adagrad-optimizer |
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Anchor Box Generation | Generate object-detection anchor boxes across a feature grid for every scale and aspect-ratio combination. | https://www.tensortonic.com/problems/anchor-box-generation |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Advantage Computation | Compute reinforcement-learning advantages by subtracting value estimates from observed returns at each timestep. | https://www.tensortonic.com/problems/compute-advantage |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Discounted Returns | Compute discounted reinforcement-learning returns backward through a reward sequence using a discount factor. | https://www.tensortonic.com/problems/discount-returns |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| ε-Greedy Action Selection | Select a reinforcement-learning action with epsilon-greedy exploration using action values and controlled randomness. | https://www.tensortonic.com/problems/epsilon-greedy |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Generalized Advantage Estimation | Compute generalized advantage estimates backward through rewards, values, discounting, and trace decay. | https://www.tensortonic.com/problems/gae-computation |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| Gradient Clipping (Global Norm) | Clip a NumPy gradient array by its global L2 norm while preserving direction when scaling is required. | https://www.tensortonic.com/problems/gradient-clipping |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Build a Mini GRU Cell (Forward Pass) | Implement a GRU cell forward pass with reset, update, and candidate gates for one sequence timestep. | https://www.tensortonic.com/problems/gru-cell-forward |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max Pooling Forward | Apply 2D max pooling to a numeric matrix using a configurable square window and stride. | https://www.tensortonic.com/problems/maxpool-forward |
| Monte Carlo Policy Evaluation | Estimate state values from complete Monte Carlo episodes by averaging discounted returns for each visited state. | https://www.tensortonic.com/problems/mc-policy-evaluation |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Policy Gradient Loss | Compute policy-gradient loss from selected action probabilities and advantage estimates with stable logarithms. | https://www.tensortonic.com/problems/policy-gradient-loss |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Prioritized Experience Replay | Compute prioritized replay sampling probabilities and normalized importance weights from transition priorities. | https://www.tensortonic.com/problems/priority-replay-sample |
| Tabular Q-Learning (Single Update) | Perform one tabular Q-learning update from reward, discount, learning rate, and the best next-state value. | https://www.tensortonic.com/problems/q-learning-update |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| Replay Buffer Sample | Sample a reproducible mini-batch of transitions from a replay buffer without modifying stored experience. | https://www.tensortonic.com/problems/replay-buffer-sample |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| RNN Step Backward (Vanilla RNN) | Backpropagate through one vanilla RNN timestep to compute input, hidden-state, weight, and bias gradients. | https://www.tensortonic.com/problems/rnn-step-backward |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| SARSA Update | Perform one on-policy SARSA action-value update from the observed reward and next selected action. | https://www.tensortonic.com/problems/sarsa-update |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement a Simple CNN Layer (NumPy) | Implement a NumPy CNN layer forward pass with batched valid convolution across channels and bias addition. | https://www.tensortonic.com/problems/simple-cnn-layer |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| One-Step TD Value Update | Perform one temporal-difference value update from reward, discount, next-state value, and learning rate. | https://www.tensortonic.com/problems/td-value-update |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Word Count Dictionary | Count token occurrences in text and return a dictionary mapping each distinct word to its frequency. | https://www.tensortonic.com/problems/word-count-dict |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Activation Functions | Implement ReLU, sigmoid, tanh, Leaky ReLU, GELU, and Swish with their analytical derivatives. | https://www.tensortonic.com/study-plans/cracking-dl/dl-activation-functions |
| Batch Normalization | Implement batch normalization for training and inference, including batch statistics and running-statistic updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-batch-normalization |
| Dropout | Apply inverted dropout from a supplied binary mask during training and preserve the input unchanged during evaluation. | https://www.tensortonic.com/study-plans/cracking-dl/dl-dropout |
| Multi-Layer Perceptron (Forward Pass) | Implement the forward pass of a multi-layer perceptron (MLP) with arbitrary depth and width. | https://www.tensortonic.com/study-plans/cracking-dl/dl-forward-pass |
| Loss Functions | Implement MSE, binary cross-entropy, categorical cross-entropy, and Huber losses from supplied predictions and targets. | https://www.tensortonic.com/study-plans/cracking-dl/dl-loss-functions |
| Mini-Batch Training Loop | Train a NumPy multilayer perceptron over ordered mini-batches with forward passes, backpropagation, and SGD updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-mini-batch-training |
| Perceptron | Train a binary perceptron from zero-initialized weights using ordered samples, step predictions, and error-correction updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-perceptron |
| Weight Initialization | Compute per-layer NumPy weight initialization parameters from network dimensions and the selected initialization method. | https://www.tensortonic.com/study-plans/cracking-dl/dl-weight-initialization |
| Lasso Regression | Implement Lasso regression with gradient descent, an L1 subgradient penalty on weights, and an unregularized bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-lasso-regression |
| Linear Regression from Scratch | Train linear regression from scratch with mean squared error gradients for weights and bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-linear-regression-from-scratch |
| Logistic Regression from Scratch | Train binary logistic regression from scratch using sigmoid probabilities, cross-entropy gradients, and gradient descent. | https://www.tensortonic.com/study-plans/cracking-ml/ml-logistic-regression |
| Ridge Regression | Train Ridge regression with gradient descent, L2-regularized weights, and an unregularized bias term. | https://www.tensortonic.com/study-plans/cracking-ml/ml-ridge-regression |
| Softmax Regression | Train multiclass softmax regression with stable probabilities, one-hot targets, cross-entropy gradients, and gradient descent. | https://www.tensortonic.com/study-plans/cracking-ml/ml-softmax-regression |
| Bellman Expectation Equation | The Bellman expectation equation is the cornerstone of policy evaluation in reinforcement learning. | https://www.tensortonic.com/study-plans/cracking-rl/rl-bellman-expectation-equation |
| Bellman Optimality Equation | Apply one Bellman optimality backup by maximizing expected immediate reward plus discounted next-state value. | https://www.tensortonic.com/study-plans/cracking-rl/rl-bellman-optimality-equation |
| Discounted Returns | Compute reverse-time discounted returns from a reward sequence using a configurable discount factor and terminal bootstrap. | https://www.tensortonic.com/study-plans/cracking-rl/rl-discounted-returns |
| Double DQN Target | Compute Double DQN targets by selecting actions with the online network and evaluating them with the target network. | https://www.tensortonic.com/study-plans/cracking-rl/rl-double-dqn-target |
| DQN Loss with Target Network | Compute DQN targets and mean squared temporal-difference loss with terminal masking and a separate target network. | https://www.tensortonic.com/study-plans/cracking-rl/rl-dqn-loss-with-target-network |
| Dueling DQN Decomposition | Dueling DQN splits the action-value function into a state value head V(s) and an advantage head A(s, a), then aggregates them back into Q(s, a). | https://www.tensortonic.com/study-plans/cracking-rl/rl-dueling-dqn-decomposition |
| Epsilon-Greedy Action Selection | Choose actions with epsilon-greedy exploration, deterministic greedy tie handling, and controlled random sampling. | https://www.tensortonic.com/study-plans/cracking-rl/rl-epsilon-greedy-action-selection |
| Expected SARSA | Apply Expected SARSA updates using the epsilon-greedy expectation over next-state action values. | https://www.tensortonic.com/study-plans/cracking-rl/rl-expected-sarsa |
| Experience Replay Buffer | Implement a fixed-capacity circular replay buffer with deterministic overwrites and sampled transition batches. | https://www.tensortonic.com/study-plans/cracking-rl/rl-experience-replay-buffer |
| LinUCB Contextual Bandit | LinUCB extends the UCB algorithm to contextual bandits by assuming the expected reward of each arm is a linear function of the context. | https://www.tensortonic.com/study-plans/cracking-rl/rl-linucb-contextual-bandit |
| Monte Carlo Policy Evaluation | Estimate policy state values from complete sampled returns without requiring an environment transition model. | https://www.tensortonic.com/study-plans/cracking-rl/rl-monte-carlo-policy-evaluation |
| Policy Iteration | Policy iteration solves a finite Markov Decision Process by alternating two steps until the greedy policy stops changing. | https://www.tensortonic.com/study-plans/cracking-rl/rl-policy-iteration |
| Prioritized Experience Replay | Sample replay items from priority-weighted probabilities and compute normalized importance-sampling weights. | https://www.tensortonic.com/study-plans/cracking-rl/rl-prioritized-experience-replay |
| Q-Learning Update | Apply the off-policy Q-learning update using the maximum next-state action value and terminal handling. | https://www.tensortonic.com/study-plans/cracking-rl/rl-q-learning-update |
| SARSA Update | Apply the on-policy SARSA temporal-difference update from the current transition and next selected action. | https://www.tensortonic.com/study-plans/cracking-rl/rl-sarsa-update |
| Softmax (Boltzmann) Exploration | Convert action values into a numerically stable Boltzmann exploration distribution controlled by temperature. | https://www.tensortonic.com/study-plans/cracking-rl/rl-softmax-exploration |
| TD(0) Value Update | Apply one TD(0) state-value update using immediate reward, discounted next-state value, and learning rate. | https://www.tensortonic.com/study-plans/cracking-rl/rl-td-zero-update |
| Thompson Sampling (Beta-Bernoulli) | Thompson sampling for a Bernoulli multi-armed bandit keeps a Beta posterior over each arm's success probability. | https://www.tensortonic.com/study-plans/cracking-rl/rl-thompson-sampling-beta-bernoulli |
| UCB1 for Multi-Armed Bandits | In a multi-armed bandit, an agent must repeatedly choose one of K arms and only observes the reward of the arm it pulls. | https://www.tensortonic.com/study-plans/cracking-rl/rl-ucb1-multi-armed-bandit |
| Value Iteration | Run value iteration to convergence for a finite MDP, then extract the greedy optimal policy with stable tie handling. | https://www.tensortonic.com/study-plans/cracking-rl/rl-value-iteration |
| Continuity of Activation Functions | Classify continuity and nondifferentiable points for common activation functions at supplied scalar inputs. | https://www.tensortonic.com/study-plans/math-calculus/calculus-activation-continuity |
| Limit of a Learning Rate Schedule | Evaluate the long-run limit of an inverse-time learning-rate schedule from its initial rate and decay constant. | https://www.tensortonic.com/study-plans/math-calculus/calculus-lr-schedule-limit |
| Cholesky Decomposition | Factor a symmetric positive-definite NumPy matrix into a lower-triangular matrix and its transpose. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-cholesky |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with explicit handling for zero-norm inputs. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-cosine-similarity |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-dot-product |
| Eigendecomposition | Compute eigenvalues and aligned eigenvectors for a square NumPy matrix and verify the decomposition relationship. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-eigendecomposition |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors from the square root of summed squared differences. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-euclidean-distance |
| Gram-Schmidt Orthogonalization | Given k linearly independent vectors in R^n, the Gram-Schmidt process builds an orthonormal basis that spans exactly the same space. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-gram-schmidt |
| Hadamard Product | Compute elementwise multiplication between two same-shaped NumPy matrices to produce their Hadamard product. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-hadamard-product |
| Least Squares Solution | Solve a full-column-rank least-squares system and return the coefficient vector minimizing residual norm. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-least-squares |
| Linear Combination | Compute a weighted linear combination of equal-length NumPy vectors using one aligned scalar coefficient per vector. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-linear-combination |
| Low-Rank Approximation | Construct the best rank-k matrix approximation from truncated singular values and singular vectors. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-low-rank-approx |
| LU Decomposition | Factor a square matrix into lower and upper triangular matrices using the specified no-pivot LU procedure. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-lu-decomposition |
| Mahalanobis Distance | Compute Mahalanobis distance between a point and a distribution using its mean and covariance matrix. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-mahalanobis |
| Matrix Determinant | Compute the determinant of a square NumPy matrix as a scalar measure of invertibility and volume scaling. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-determinant |
| Matrix Multiply | Multiply compatible NumPy matrices and preserve the dtype produced by NumPy type-promotion rules. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-multiply |
| Matrix Rank | Compute the rank of a rectangular NumPy matrix from its number of linearly independent directions. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-rank |
| Matrix Trace | Compute the trace of a square NumPy matrix by summing its main-diagonal elements with numeric dtype support. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-trace |
| Matrix Transpose | Transpose a rectangular NumPy matrix by swapping its row and column axes without changing element values. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-transpose |
| Matrix-Vector Multiply | Multiply a NumPy matrix by a compatible vector, producing one row-wise dot product per output element. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-vector-multiply |
| Orthogonal Projection Matrix | Construct the orthogonal projection matrix onto the column space of a full-column-rank matrix. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-orthogonal-projection |
| Outer Product | Compute the NumPy outer product of two vectors as a matrix containing every pairwise element multiplication. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-outer-product |
| PCA from Scratch | Implement PCA from scratch by centering data, decomposing covariance, and projecting onto leading components. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-pca-from-scratch |
| Moore-Penrose Pseudoinverse | Compute the Moore-Penrose pseudoinverse of rectangular or singular matrices using singular-value decomposition. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-pseudoinverse |
| QR Decomposition | Compute its QR decomposition: factor A into an orthogonal matrix Q and an upper triangular matrix R. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-qr-decomposition |
| RBF Kernel Matrix | Compute the pairwise radial-basis-function kernel matrix from sample vectors and a positive bandwidth. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-rbf-kernel |
| Scaled Dot-Product Attention | Implement the scaled dot-product attention mechanism from the Transformer architecture. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-scaled-attention |
| Solve Linear System | Solve an invertible square linear system for the unique vector satisfying the matrix equation. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-solve-linear-system |
| SVD Components | Compute singular values and aligned singular vectors for a possibly rank-deficient matrix using NumPy. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-svd-components |
| Vector Norms | Compute L1, L2, and infinity norms for a one-dimensional NumPy vector and return them in a float64 array. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-vector-norms |
| Vector Projection | The vector projection of u onto v is the component of u that lies exactly along the direction of v. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-vector-projection |
| Whitening Transform | Center and whiten a data matrix so the transformed features have zero mean and identity covariance. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-whitening |
| Aggregation Functions | Compute selected NumPy aggregation functions globally or along a requested axis using float64 values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-aggregation |
| Angle Features | Return a float64 array where row 0 contains the sine values, row 1 the cosine values, and row 2 the tangent values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-angle-features |
| Arange and Linspace | Generate a one-dimensional NumPy sequence using either step-based arange or count-based linspace semantics. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-arange-linspace |
| Basic Indexing | Extract a rectangular NumPy subarray with row and column slice boundaries using standard basic indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-basic-indexing |
| Boolean Masking | Build three filtered views of a 2D array: an element-level boolean mask, rows kept when any element exceeds a threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-boolean-masking |
| Column Scaling | Scale every column of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-col-scaling |
| Concat and Correlate | Concatenate two 2-D arrays row-wise and return a (3, n, n) stack of Pearson correlation matrices: one for each input and one for the combined data. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-concat-correlate |
| Create Arrays from Lists | Create NumPy arrays from Python lists with the requested dtype and return their values, shape, dimensions, and element count. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-create-array |
| Fancy Indexing | Convert the data to float64 and return the array formed by selecting elements along that axis using integer array indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-fancy-indexing |
| Filter and Extract | Implement Filter and Extract, and apply a boolean mask to select values strictly greater than threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-filter-extract |
| Mutation Trap | Extract an independent NumPy row copy, mutate it safely, and verify that the original array remains unchanged. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-mutation-trap |
| Normalized Difference | Use two 2D arrays a and b of the same shape and a scalar range [lo, hi], clip both arrays to [lo, hi], rescale each to [0, 1]. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-norm-diff |
| Norm-Gated Linear Transform | Compute the linear transform Z = X @ W, then zero out every row of Z whose L2 norm is strictly below the threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-norm-gate |
| Normalize Columns | Standardize each NumPy matrix column by subtracting its mean and dividing by its population standard deviation. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-normalize-columns |
| Outer Sum | Compute the broadcasted outer sum of two NumPy vectors without loops, supporting different lengths and numeric values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-outer-sum |
| Pairwise Differences | Implement Pairwise Differences, and compute the pairwise difference matrix without any Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-pairwise-diff |
| Quantize and Frame | Apply floor, ceiling, and nearest rounding to a NumPy matrix, then add a zero-valued border around each result. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-quantize-frame |
| Random Array Generation | Generate seeded float64 NumPy arrays from either a uniform or standard normal distribution. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-random-arrays |
| Reshaping Arrays | Transform a float64 NumPy array with flattening, transposition, or a validated target shape. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-reshape |
| Row Extremes | Implement Row Extremes, using np.argmax(axis=1) to find the column index of the maximum value in each row. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-row-extremes |
| Row Scaling | Scale every row of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-row-scaling |
| Sort and Argsort | Return NumPy values sorted along a selected axis together with the indices that produce the same ordering. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-sort-argsort |
| Tile and Diff | Tile a 2-D array vertically and return the tiled result alongside its row-wise finite differences, packed as a (2, m·reps, n) float64 array. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-tile-diff |
| Winsorize | Winsorization clips extreme values in each column to percentile-based bounds, a standard technique for suppressing outliers in ML preprocessing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-winsorize |
| Zeros and Ones | Create a two-dimensional float64 NumPy array of a requested shape filled entirely with zeros or ones. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-zeros-ones |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/kingphito)
<!-- tensortonic:end -->

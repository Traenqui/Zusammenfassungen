#import "../../template_zusammenf.typ": *

= Deep Neural Networks (DNN) - Recap

_N-Grams_: Fixed-size context windows → sparse reps, limited generalization; cannot capture long-range deps beyond window size $n$.

_Hyperparameters_:
Learning rate, number of epochs, batch size, architecture choices #hinweis[(layers, neurons per layer, activations)], regularization #hinweis[(L1, L2, dropout)].

== Training a Neural Network
*Stochastic Gradient Descent (SGD):*
Update weights using single data point
$ w_(t+1) <- w_t - alpha partial/(partial w) log(p(x_t)) $

*Batch Gradient Descent:*
Update weights using all N data points
$ w_(t+1) <- w_t - alpha (1/N) sum^N_(i=1) partial/(partial w) log(p(x_i)) $
More stable but computationally expensive

*Mini-batch Gradient Descent:*
Updates using m data points
$ w_(t+1) <- w_t - alpha (1/m) sum^m_(i=1) partial/(partial w) log(p(x_i)) $
Balances stability and computational efficiency

== Activation Functions
#table(
  columns: (1.2fr, 1fr, 1.5fr, 1.5fr),
  table.header([*Function*], [*Formula*], [*Pros*], [*Cons*]),
  [ReLU], [$f(x) = max(0, x)$], [Fast, avoids vanishing gradients, default for hidden layers], ["Dying ReLU" where neurons become inactive],
  [Sigmoid], [$f(x) = 1/(1 + e^(-x))$], [Outputs in $(0, 1)$, for binary classification], [Vanishing gradient, not zero-centered],
  [Tanh], [$f(x) = (e^x - e^(-x))/(e^x + e^(-x))$], [Outputs in $(-1, 1)$, zero-centered], [Vanishing gradient for large inputs],
  [Leaky ReLU], [$f(x) = max(alpha x, x)$\ #hinweis[($alpha approx 0.01$)]], [Prevents dead neurons, allows small gradient], [Introduces hyperparameter $alpha$],
  [Softmax], [$f(x_i) = e^(x_i) / sum_j e^(x_j)$], [Outputs probability distribution], [Only suitable for output layer],
)

== Loss Functions
#table(
  columns: (1.2fr, 1.8fr, 1fr),
  table.header([*Loss Function*], [*Formula*], [*Use Case*]),
  [Mean Squared Error\ (MSE)], [$L = 1/n sum_(i=1)^n (y_i - hat(y)_i)^2$], [Regression],
  [Binary Cross-Entropy], [$L = -1/n sum_(i=1)^n [y_i log(hat(y)_i) + (1-y_i)log(1-hat(y)_i)]$], [Binary\ classification],
  [Cross Entropy], [Measures difference between true and predicted probability distributions], [Multi-class],
  [Categorical Cross-Entropy], [$L = -sum_(i=1)^n sum_(c=1)^C y_(i,c) log(hat(y)_(i,c))$], [One-hot encoded\ multi-class],
)

*Rule of thumb:* ReLU for hidden layers, MSE for regression, BCE for binary classification, Cross Entropy for multi-class classification.

== Likelihood
*Likelihood:* $L(theta) = p(x | theta)$ - Probability of observed data given model parameters.

*Log-Likelihood:* $log L(theta) = log p(x | theta) = sum_i log p(x_i | theta)$
Simplifies optimization by converting products into sums. Higher → better

*Negative Log-Likelihood (NLL):* $"NLL" = -sum_i log p(x_i | theta)$
Loss minimized during training; equivalent to maximizing log-likelihood. Lower → better

== Convolutional Neural Networks (CNN)

=== Parameters in a Convolutional Layer
The number of trainable parameters depends on:
- kernel size $(k_h times k_w)$
- number of input channels $C_"in"$
- number of filters (output channels) $C_"out"$
- one bias per filter

Total parameters: $(k_h times k_w times C_"in") times C_"out" + C_"out"$

*Example:* Input: $28 times 28$ grayscale $(C_"in" = 1)$, Filters: 10, Kernel: $3 times 3$\
Parameters: $(3 times 3 times 1) times 10 + 10 = 100$

*Key concepts:*
- *Convolution:* slide kernel/filter over input to detect local patterns
- *Padding:* add borders to maintain spatial dimensions
  #hinweis[(SAME padding: output size = input size; VALID: no padding)]
- *Stride:* step size of kernel movement
  $ "output size" = ((n + 2p - f)/s) + 1 $
  where $n$ = input size, $p$ = padding, $f$ = filter size, $s$ = stride
- *Pooling:* downsample feature maps
  #hinweis[(Max pooling: maximum value; Average pooling: mean)]

== Evaluation Metrics
#table(
  columns: (1fr, 1.5fr, 1.5fr),
  table.header([*Metric*], [*Formula*], [*When to Use*]),
  [Accuracy], [$("TP" + "TN")/"n"$], [Balanced datasets],
  [Precision], [$"TP"/("TP" + "FP")$], [When false positives matter],
  [Recall], [$"TP"/("TP" + "FN")$], [When false negatives matter],
  [F1 Score], [$(2 dot "Precision" dot "Recall")/("Precision" + "Recall")$], [Both precision and recall important],
)

*Rule of thumb:* Use Accuracy for balanced data, Precision when false positives matter, Recall when false negatives matter, and F1 when both are important.

== Regularization Techniques
*L1 Regularization (Lasso):*
$ L = L_"original" + lambda sum_i |w_i| $
Promotes sparsity #hinweis[(many weights become exactly zero)]

*L2 Regularization (Ridge):*
$ L = L_"original" + lambda sum_i w_i^2 $
Encourages small weights, prevents overfitting

*Dropout:*
Randomly deactivate neurons during training with probability $p$ #hinweis[(typically $p = 0.5$)].
Forces network to learn robust features.

*Early Stopping:*
Monitor validation loss and stop when it stops improving.
#import "../../template_zusammenf.typ": *

= Deep Neural Networks (DNN) — Recap

_N-grams_: Fixed-size context window → sparse representations, limited generalization; cannot model dependencies beyond window size $n$.

_Hyperparameters_: learning rate $alpha$, epochs, batch size, architecture (layers/width), activation functions, regularization #hinweis[(L1/L2, dropout)], optimizer settings.

== Training a Neural Network

Goal: learn parameters $theta$ by minimizing a loss $L(theta)$ (often NLL / cross-entropy).


*Stochastic Gradient Descent (SGD):* Updates weights using single data point

$ w_(t+1) <- w_t - alpha (partial)/(partial w) log(p(x_t)) $

*Batch GD:* Updates using all N data points

$ w_(t+1) <- w_t - alpha (1/N) sum_(i=1)^N (partial)/(partial w) log(p(x_i)) $

*Mini-Batch GD:* Updates using m data points

$ w_(t+1) <- w_t - alpha (1/m) sum_(i=1)^m (partial)/(partial w) log(p(x_i)) $

$w_t$ = weights at time $t$, $alpha$ = learning rate, $log(p(x_i))$ = log likelihood of point $x_i$, $(partial)/(partial w)$ = gradient w.r.t the weights


#hinweis[
If you maximize log-likelihood directly (gradient ascent):
$theta <- theta + alpha nabla_theta log p_theta(x_i) $.
Minimizing NLL is equivalent: $L(x_i; theta) = -log p_theta(x_i)$.
]

== Activation Functions

#table(
  columns: (1.2fr, 1.4fr, 2.8fr),
  align: (left, left, left),

  [*Function*], [*Definition*], [*Notes*],

  [ReLU], [$max(0, x)$],
  [Default for hidden layers; can produce “dead” units if always negative.],

  [Leaky ReLU], [$max(alpha x, x)$],
  [$alpha approx 0.01$; mitigates dead ReLUs.],

  [Sigmoid], [$sigma(x)= frac(1, 1 + exp(-x))$],
  [Outputs $(0,1)$; saturates → vanishing gradients; good for binary probability outputs.],

  [tanh], [$tanh(x)$],
  [Outputs $(-1,1)$; zero-centered but saturates.],

  [Softmax], [$"softmax"(z)_k = frac(exp(z_k), sum_(j=1)^K exp(z_j))$],
  [Multi-class probabilities; used with cross-entropy.],
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

== Likelihood / NLL (Connection to Probabilities)

Likelihood of dataset $D={x_1, ..., x_N}$ under model $p_theta$: $L(theta) = p_theta(D) = "prod"_(i=1)^N p_theta(x_i) $

_Log-likelihood_: $log L(theta) = sum_(i=1)^N log p_theta(x_i) $

_Negative log-likelihood (NLL)_: $"NLL"(theta) = - sum_(i=1)^N log p_theta(x_i) $

== CNN Quick Facts

*Conv-layer parameter count* (kernel $k_h times k_w$, input channels $C_in$, output channels $C_"out"$):
$ "#params" = (k_h k_w C_in) C_"out" + C_"out" $

*Output spatial size* (input $n$, padding $p$, filter size $f$, stride $s$): $n_"out" = floor( frac(n + 2p - f, s) ) + 1 $

Pooling: max/avg; reduces spatial size; adds invariance.

== Evaluation Metrics

#table(
  columns: (1fr, 1.5fr, 1.5fr),
  table.header([*Metric*], [*Formula*], [*When to Use*]),
  [Accuracy], [$("TP" + "TN")/"n"$], [Balanced datasets],
  [Precision], [$"TP"/("TP" + "FP")$], [When false positives matter],
  [Recall], [$"TP"/("TP" + "FN")$], [When false negatives matter],
  [F1 Score], [$(2 dot "Precision" dot "Recall")/("Precision" + "Recall")$], [Both precision and recall important],
)

#hinweis[
Rules of thumb: Accuracy (balanced), Precision (FP costly), Recall (FN costly), F1 (tradeoff).
]

== Regularization Techniques

_L1 regularization (Lasso)_: $L = L_"orig" + lambda sum_i abs(w_i) $
Promotes sparsity #hinweis[(many weights become exactly zero)].

_L2 regularization (Ridge / weight decay)_: $L = L_"orig" + lambda sum_i w_i^2 $
Encourages small weights; reduces overfitting.

*Dropout:*
Randomly deactivate units during training with probability $p$.
Typical ranges: $p approx 0.1$–$0.5$ depending on layer.
Forces robust feature learning.

*Early stopping:*
Monitor validation loss/metric; stop when it stops improving (prevents overfitting).

## Natural Gradients in Tensorflow

So I recently started learning deep reinforcement learning, and decided to make an open source Deep RL framework called ReiLS.

So I went ahead and implemented a couple of popular actor-critic methods like DDPG, A3C and the more recent PPO, and soon turned my attention to TRPO.

The difficulty with TRPO is that it uses natural gradients, as opposed to regular gradients.

Natural gradients are pretty freaking cool, tbh.
Normally we assume the parameter space `S` to Euclidean with an orthonormal coordinate system. So that's the normal 3D space we're familiar with. Using regular gradients here would be ideal.

However, when `S` is a curved manifold, there is no orthonormal coordinate system. This is particularly when we're in non-Euclidean space, which is what we deal with in most neural network. So the gradients we calculate aren't the true gradients.

So let `L(w)` be the loss function defined in `S`, the direction of steepest descent of `L(w)` at `w` is defined as the vector `dw` that minimizes `L(w+dw)`, where `dw` has a fixed length.

According to this one baller dude called Riemann, the steepest direction is given by

<center> -&#916;<sub>nat</sub>L(w) = -G<sup>-1</sup> &#916;L(w)


where &#916;<sub>nat</sub> stands natural gradient, &#916; is conventional gradient, and `G` is a matrix called the Riemannian metric.

Note that `G` depends on the `w`, and so is location dependant.

Intuitively, the Riemannian metric tensor describes how the geometry of a manifold affects a differential patch, `dw`, at the point `w`. The length of a line between two points on `dw` is the distance between them. The Riemannian metric tensor either stretches or shrinks that line and the resulting length is the distance between the two points on the manifold.

When the space is Euclidean, G is an identity matrix, so

<center> &#916;<sub>nat</sub>L(w) = &#916;L(w)

This suggests that the gradient descent algorithm should be

<center> w<sub>t+1</sub> = w<sub>t</sub> - &alpha; &#916;<sub>nat</sub>L(w<sub>t</sub>)

where &alpha; is the learning rate

For neural networks, `G`, the Riemannian metric is given by the Fisher Information Matrix.

### Fisher Information

Fisher information is the second derivative of KL divergence

<center>
F<sub>&theta;</sub> = &#916;<sub>&theta;'</sub><sup>2</sup> D(&theta;'||&theta;)|<sub>&theta;'=&theta;</sup>
</center>



<center>
F<sub>&theta;</sub> = &#916;<sub>&theta;</sub><sup>2</sup> D(&theta;||&theta;')|<sub>&theta;'=&theta;</sup>
</center>

Where D(&phi;||&beta;) is the KL divergence between the output distributions of the same model parameterised by &phi; and &beta;, where both belong to the same parameter space.

Both directions of KL divergence have the same  second-order derivative at the point where the distributions match, so locally KL divergence is sort've symmetric.

Using second-order Taylor expansion, we can write

<center>
D(&theta;'||&theta;) = 0.5 * (&theta;' - &theta;)<sup>T</sup> F <sub>&theta;</sub>(&theta;' - &theta;)
</center>
<sub><sub>We assume &theta;' - &theta; is small, else the approximation won't work.</sub></sub>

Since KL divergence is similar to distance between two distributions, Fisher Information gives you the *local* distance between distributions. Intuitively, it gives the change in the distribution for a small change in parameters. This is why we can use it as `G`.

##### Tensorflow Code

Let's do MNIST classification using Natural gradients

This a Python class for Categorical probability distribution, used for discrete classes.

    class CategoricalPd:
        def __init__(self, logits):
            self.logits = logits

        def mode(self):
            return tf.argmax(self.logits, axis = -1)

        def kl(self, other):
            a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims = True)
            a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims = True)

            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)

            z0 = tf.reduce_sum(ea0, axis = -1, keepdims = True)
            z1 = tf.reduce_sum(ea1, axis = -1, keepdims = True)

            p0 = ea0 / z0

            s0 = a0 - tf.log(z0)
            s1 = a1 - tf.log(z1)

            return tf.reduce_sum(p0 * (s0 - s1), axis = -1)

        def self_kl(self):
            other = CategoricalPd(tf.identity(self.logits))
            other.logits = tf.stop_gradient(other.logits)

            return self.kl(other)

Now we initialize our placeholder for images and corresponding labels:

    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])
	lr = tf.placeholder(tf.float32, shape=())

Let's make a simple single layer neural network, for which we'll need the weight vector:

    W = tf.get_variable('w', shape=[7840])

The matrix multiplication step and loss calculation:

    output = tf.matmul(X, tf.reshape(W, [784, 10]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))

As you can see above, we're using the cross entropy loss.

Time to calculate the natural gradients. First we need the conventional gradients:

    grad_cov = tf.gradients(loss, W)

For finding the Fisher Information Metric, we'll need the KL of our output, let's make use of `CategoricalPd` class above for this.

    pd = CategoricalPd(output)
    kl = tf.reduce_mean(pd.self_kl())

We can easily find the second order derivative of of `kl` using Tensorflow:

    fish = tf.hessians(kl, W)

We can now calculate the natural gradients as mentioned above

    fish_inv = tf.matrix_inverse(fish)
    grad_true = tf.matmul(grad_cov, fish_inv[0])
    grad_true = tf.reshape(grad_true, [-1])

We need to do some reshaping so the vectors' sizes match.

We can now use a simple gradient descent optimizer to train our network:

    opt = tf.train.GradientDescentOptimizer(lr)
    train = opt.apply_gradients([(grad_true, W)])

You can now create a `tf.Session` can run the `train` op feeding values into `X` and `Y`. But there's a problem.

Calculating the hessian is a expensive, since for `n` params, you'll be calculating `n` gradients. If you check, `fish` is matrix of size `(7840, 7840)`, and trying to find its inverse is computationally expensive, since matrix inverting is an O(n<sup>3</sup>) algorithm, so for our matrix, the number of operations is around ~10<sup>12</sup>. This is for the simplest single layer neural network. Obviously the naive approach will not work for deep learning models without some other clever algorithm.

Turns out the above code is too slow for a K80, so I downsampled all the images to `(14, 14)`, reducing the size of `w` by a factor of 4, and the size of the fisher by a factor of 16. So it should be a lot faster.

I ran into another problem where `fish` was a singular matrix, so I couldn't calculate it's inverse, so I figured I'd use the Moore-Penrose pseudo-inverse, or for the NumPy fanatics -- `np.linalg.pinv`, but suprise suprise, Tensorflow does not have an implementation of pseudo-inverse. After going through the documentation for [Tensorflow's linalg module](https://www.tensorflow.org/api_docs/python/tf/linalg), I came across `tf.svd`, Tensorflow's GPU implementation of [Singular Value Decomposition](https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf), so I wrote my own version to pseudo-invert my matrix and then matmul with `grad_cov` to find `grad_true`

	def pinv(A, b, reltol=1e-6):
	  s, u, v = tf.svd(A)
	
	  atol = tf.reduce_max(s) * reltol
	  s = tf.boolean_mask(s, s > atol)
	  s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))
	
	  # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
	  return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))

I finally could train the simple one layer network with a learning rate of `1e-1` and batch size of `32`, here's the convergence graph along with vanilla gradients thrown in to show that all the work is worth something. As you can see the natural gradient descent(orange) reaches the same loss as conventional gradient descent(blue) is much few iterations.

![alt text][naive-plot]

Calculating the hessian and its inverse is shown to be expensive because each iteration of natural gradient descent took around 30 seconds. As compared to regular gradient descent, where I did 1000 iterations in less than 3 seconds. Clearly, we need a more efficient way to do natural gradient descent, one of the most popular ways is to use conjugate descent.

### Conjugate Gradient Descent

[naive-plot]: https://github.com/Squadrick/natural-gradients/blob/master/results/naive-results.png "Naive descent comparision"

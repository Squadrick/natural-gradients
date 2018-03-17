import tensorflow as tf
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("~", one_hot=True)
print("done loading data")

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
        logits = tf.identity(self.logits)
        other = CategoricalPd(tf.stop_gradient(logits))

        return self.kl(other)
    
x_in = tf.placeholder(tf.float32, shape=[None, 14, 14])
X = tf.reshape(x_in, shape=[-1, 14*14])

Y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.get_variable('wwww', shape=[14*14*10])

output = tf.matmul(X, tf.reshape(W, [14*14, 10]))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))

grad_cov = tf.gradients(loss, W)

pd = CategoricalPd(output)
kl = tf.reduce_mean(pd.self_kl())
lr = tf.placeholder(tf.float32, ())
fish = tf.hessians(kl, W)

def pinv(A, b, reltol=1e-6):
  s, u, v = tf.svd(A)

  atol = tf.reduce_max(s) * reltol
  s = tf.boolean_mask(s, s > atol)
  s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))

  # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
  return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))

#fish_inv = tf.matrix_inverse(fish)
#grad_true = tf.matmul(grad_cov, fish_inv[0])
grad_true = pinv(fish[0], grad_cov)
grad_true = tf.reshape(grad_true, [-1])
opt = tf.train.GradientDescentOptimizer(lr)
l = [(grad_true, W)]
train = opt.apply_gradients(l)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

def util_function_img(images):
    smaller_images = []
    for img in images:
        img = np.reshape(img, (28, 28))
        img = cv2.resize(img, (14, 14))
        smaller_images.append(img)
    return np.asarray(smaller_images)

def util_function_lab(labels):
    smaller_labs = []
    for lab in labels:
        new_lab = [0, 0]
        new_lab[np.argmax(lab)%2] = 1
        smaller_labs.append(new_lab)
    return np.array(smaller_labs)

for i in range(10000):
    x, y = mnist.train.next_batch(16)
    x = util_function_img(x)
    l, _ = sess.run([loss, train], feed_dict={x_in:x, Y:y, lr:1e-1})
    print(l)

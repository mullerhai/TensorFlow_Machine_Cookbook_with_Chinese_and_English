import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

############# Gate1
# Start Graph Session
sess = tf.Session()

a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)
multiplication = tf.multiply(a, x_data)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.subtract(multiplication, 50.))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Run loop across gate
print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))




############# Gate2
# Start a New Graph Session
ops.reset_default_graph()
sess = tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

two_gate = tf.add(tf.multiply(a, x_data), b)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.subtract(two_gate, 50.))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Run loop across gate
print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val, b_val = (sess.run(a), sess.run(b))
    two_gate_output = sess.run(two_gate, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))


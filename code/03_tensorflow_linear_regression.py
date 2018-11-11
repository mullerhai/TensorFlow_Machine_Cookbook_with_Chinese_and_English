# # 1 用TensorFlow求逆矩阵
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# sess = tf.Session()
#
# x_vals = np.linspace(0,10,100)
# y_vals = x_vals + np.random.normal(0,1,100)
#
# x_vals_column = np.transpose(np.matrix(x_vals))
# ones_column = np.transpose(np.matrix(np.repeat(1,100)))
# A = np.column_stack((x_vals_column,ones_column))
# b = np.transpose(np.matrix(y_vals))
#
# A_tensor = tf.constant(A)
# b_tensor = tf.constant(b)
#
# tA_A = tf.matmul(tf.transpose(A_tensor),A_tensor)
# tA_A_inv = tf.matrix_inverse(tA_A)
# product = tf.matmul(tA_A_inv,tf.transpose(A_tensor))
# solution = tf.matmul(product,b_tensor)
# solution_eval = sess.run(solution)
#
# slope = solution_eval[0][0]
# y_intercept = solution_eval[1][0]
# print('slope:' + str(slope))
# print('y_intercept:' + str(y_intercept))
#
# # visualization
# best_fit = []
# for i in x_vals:
#     best_fit.append(slope*i+y_intercept)
# plt.plot(x_vals,y_vals,'o',label='Data')
# plt.plot(x_vals,best_fit,'r-',label='Best fit line',linewidth=3)
# plt.legend(loc='upper left')
# plt.show()


# 2 用TensorFlow实现矩阵分解
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# sess = tf.Session()
#
# x_vals = np.linspace(0, 10, 100)
# y_vals = x_vals + np.random.normal(0, 1, 100)
# x_vals_column = np.transpose(np.matrix(x_vals))
# ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
# A = np.column_stack((x_vals_column, ones_column))
# b = np.transpose(np.matrix(y_vals))
# A_tensor = tf.constant(A)
# b_tensor = tf.constant(b)
#
# tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
# L = tf.cholesky(tA_A)
# tA_b = tf.matmul(tf.transpose(A_tensor), b)
# sol1 = tf.matrix_solve(L, tA_b)
# sol2 = tf.matrix_solve(tf.transpose(L), sol1)
#
# solution_eval = sess.run(sol2)
# slope = solution_eval[0][0]
# y_intercept = solution_eval[1][0]
# print('slope:' + str(slope))
# print('y_intercept:' + str(y_intercept))
#
#
# # visualization
# best_fit = []
# for i in x_vals:
#     best_fit.append(slope*i+y_intercept)
# plt.plot(x_vals,y_vals,'o',label='Data')
# plt.plot(x_vals,best_fit,'r-',label='Best fit line',linewidth=3)
# plt.legend(loc='upper left')
# plt.show()

# 3 用TensorFlow实现线性回归算法
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# sess = tf.Session()
# from sklearn.datasets import load_iris
# iris = load_iris()
# x_vals = np.array([x[3] for x in iris.data])
# y_vals = np.array([y[0] for y in iris.data])
#
# learning_rate = 0.05
# batch_size = 25
# x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
# y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
# A = tf.Variable(tf.random_normal(shape=[1,1]))
# b = tf.Variable(tf.random_normal(shape=[1,1]))
#
# model_output = tf.add(tf.matmul(x_data,A),b)
#
# loss = tf.reduce_mean(tf.square(y_target - model_output))
# init = tf.global_variables_initializer()
# sess.run(init)
# my_opt = tf.train.GradientDescentOptimizer(learning_rate)
# train_step = my_opt.minimize(loss)
#
# loss_vec = []
# for i in range(100):
#     rand_index = np.random.choice(len(x_vals),size=batch_size)
#     rand_x = np.transpose([x_vals[rand_index]])
#     rand_y = np.transpose([y_vals[rand_index]])
#     sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
#     temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
#     loss_vec.append(temp_loss)
#
#     if (i+1)%25 == 0:
#         print('Step:' + str(i + 1) + ',' + 'A=' + str(sess.run(A)) + 'b=' + str(sess.run(b)))
#         print('Loss:' + str(temp_loss))
#
# [slope] = sess.run(A)
# [y_intercept] = sess.run(b)
# best_fit = []
# for i in x_vals:
#     best_fit.append(slope*i+y_intercept)
#
# plt.plot(x_vals, y_vals, 'o', label='Data Points')
# plt.plot(x_vals, best_fit, 'r-', label='Best fit line',linewidth=3)
# plt.legend(loc='upper left')
# plt.title('Sepal Length vs Pedal Width')
# plt.xlabel('Pedal Width')
# plt.ylabel('Sepal Length')
# plt.show()
#
# plt.plot(loss_vec, 'k-')
# plt.title('L2 Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('L2 Loss')
# plt.show()

# 4 理解线性回归中的损失函数
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# sess = tf.Session()
# from sklearn.datasets import load_iris
# iris = load_iris()
# x_vals = np.array([x[3] for x in iris.data])
# y_vals = np.array([y[0] for y in iris.data])
# learning_rate = 0.4
# batch_size = 25
# iterations = 50
# x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
# y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
# A = tf.Variable(tf.random_normal(shape=[1,1]))
# b = tf.Variable(tf.random_normal(shape=[1,1]))
# model_output = tf.add(tf.matmul(x_data,A),b)
#
# loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))
# loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # L1
# my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
# train_step_l1 = my_opt_l1.minimize(loss_l1)
# loss_vec_l1 = []
# for i in range(iterations):
#     rand_index = np.random.choice(len(x_vals),size=batch_size)
#     rand_x = np.transpose([x_vals[rand_index]])
#     rand_y = np.transpose([y_vals[rand_index]])
#     sess.run(train_step_l1,feed_dict={x_data:rand_x,y_target:rand_y})
#     temp_loss_l1 = sess.run(loss_l1,feed_dict={x_data:rand_x,y_target:rand_y})
#     loss_vec_l1.append(temp_loss_l1)
#     if (i+1)%25 == 0:
#         print('Step:' + str(i + 1) + ',' + 'A=' + str(sess.run(A)) + 'b=' + str(sess.run(b)))
#
# # L2
# my_opt_l2 = tf.train.GradientDescentOptimizer(learning_rate)
# train_step_l2 = my_opt_l1.minimize(loss_l2)
# loss_vec_l2 = []
# for i in range(iterations):
#     rand_index = np.random.choice(len(x_vals),size=batch_size)
#     rand_x = np.transpose([x_vals[rand_index]])
#     rand_y = np.transpose([y_vals[rand_index]])
#     sess.run(train_step_l2,feed_dict={x_data:rand_x,y_target:rand_y})
#     temp_loss_l2 = sess.run(loss_l2,feed_dict={x_data:rand_x,y_target:rand_y})
#     loss_vec_l2.append(temp_loss_l2)
#     if (i+1)%25 == 0:
#         print('Step:' + str(i + 1) + ',' + 'A=' + str(sess.run(A)) + 'b=' + str(sess.run(b)))
#
#
# plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
# plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
# plt.title('L1 and L2 Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.show()



# 5 用TensorFlow实现lasso回归和岭回归算法

# lasso回归
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
# sess = tf.Session()
# iris = datasets.load_iris()
# x_vals = np.array([x[3] for x in iris.data])
# y_vals = np.array([y[0] for y in iris.data])
# batch_size = 50
# learning_rate = 0.001
# x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
# y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
# A = tf.Variable(tf.random_normal(shape=[1,1]))
# b = tf.Variable(tf.random_normal(shape=[1,1]))
# model_output = tf.add(tf.matmul(x_data,A),b)
#
#
#
# lasso_param = tf.constant(0.9)
# heavyside_step = tf.truediv(1.,\
#                             tf.add(1.,\
#                                    tf.exp(tf.multiply(-100.,\
#                                                       tf.subtract(A,\
#                                                                   lasso_param
#                                                                  )
#                                                      )
#                                          )
#                                   )
#                             )
# regularization_param = tf.multiply(heavyside_step,99.)
# loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)),\
#               regularization_param)
#
#
# init = tf.global_variables_initializer()
# sess.run(init)
# my_opt = tf.train.GradientDescentOptimizer(learning_rate)
# train_step = my_opt.minimize(loss)
#
# loss_vec = []
# for i in range(1500):
#     rand_index = np.random.choice(len(x_vals),size=batch_size)
#     rand_x = np.transpose([x_vals[rand_index]])
#     rand_y = np.transpose([y_vals[rand_index]])
#     sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
#     temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
#     loss_vec.append(temp_loss)
#     if (i+1)%300==0:
#         print('Step:' + str(i + 1) + ',' + 'A=' + str(sess.run(A)) + 'b=' + str(sess.run(b)))
#         print('Loss:' + str(temp_loss))


# 岭回归
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
# sess = tf.Session()
# iris = datasets.load_iris()
# x_vals = np.array([x[3] for x in iris.data])
# y_vals = np.array([y[0] for y in iris.data])
# batch_size = 50
# learning_rate = 0.001
# x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
# y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
# A = tf.Variable(tf.random_normal(shape=[1,1]))
# b = tf.Variable(tf.random_normal(shape=[1,1]))
# model_output = tf.add(tf.matmul(x_data,A),b)
#
# ridge_param = tf.constant(1.)
# ridge_loss = tf.reduce_mean(tf.square(A))
# loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)
#
#
# init = tf.global_variables_initializer()
# sess.run(init)
# my_opt = tf.train.GradientDescentOptimizer(learning_rate)
# train_step = my_opt.minimize(loss)
#
# loss_vec = []
# for i in range(1500):
#     rand_index = np.random.choice(len(x_vals),size=batch_size)
#     rand_x = np.transpose([x_vals[rand_index]])
#     rand_y = np.transpose([y_vals[rand_index]])
#     sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
#     temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
#     loss_vec.append(temp_loss)
#     if (i+1)%300==0:
#         print('Step:' + str(i + 1) + ',' + 'A=' + str(sess.run(A)) + 'b=' + str(sess.run(b)))
#         print('Loss:' + str(temp_loss))


########################################### LASSO and Ridge Regression#######################################
# # This function shows how to use TensorFlow to solve LASSO or Ridge regression for y = Ax + b
# # We will use the iris data, specifically:
# #   y = Sepal Length
# #   x = Petal Width
#
# # import required libraries
# import matplotlib.pyplot as plt
# import sys
# import numpy as np
# import tensorflow as tf
# from sklearn import datasets
# from tensorflow.python.framework import ops
#
# # Specify 'Ridge' or 'LASSO'
#
# # regression_type = 'LASSO'
# regression_type = 'Ridge'
#
# # clear out old graph
# ops.reset_default_graph()
#
# # Create graph
# sess = tf.Session()
#
# ###
# # Load iris data
# ###
#
# # iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# iris = datasets.load_iris()
# x_vals = np.array([x[3] for x in iris.data])
# y_vals = np.array([y[0] for y in iris.data])
#
# ###
# # Model Parameters
# ###
#
# # Declare batch size
# batch_size = 50
#
# # Initialize placeholders
# x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
#
# # make results reproducible
# seed = 13
# np.random.seed(seed)
# tf.set_random_seed(seed)
#
# # Create variables for linear regression
# A = tf.Variable(tf.random_normal(shape=[1, 1]))
# b = tf.Variable(tf.random_normal(shape=[1, 1]))
#
# # Declare model operations
# model_output = tf.add(tf.matmul(x_data, A), b)
#
# ###
# # Loss Functions
# ###
#
# # Select appropriate loss function based on regression type
#
# if regression_type == 'LASSO':
#     # Declare Lasso loss function
#     # Lasso Loss = L2_Loss + heavyside_step,
#     # Where heavyside_step ~ 0 if A < constant, otherwise ~ 99
#     lasso_param = tf.constant(0.9)
#     heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-50., tf.subtract(A, lasso_param)))))
#     regularization_param = tf.multiply(heavyside_step, 99.)
#     loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)
#
# elif regression_type == 'Ridge':
#     # Declare the Ridge loss function
#     # Ridge loss = L2_loss + L2 norm of slope
#     ridge_param = tf.constant(1.)
#     ridge_loss = tf.reduce_mean(tf.square(A))
#     loss = tf.expand_dims(
#         tf.add(tf.reduce_mean(tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)
#
# else:
#     print('Invalid regression_type parameter value', file=sys.stderr)
#
# ###
# # Optimizer
# ###
#
# # Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.001)
# train_step = my_opt.minimize(loss)
#
# ###
# # Run regression
# ###
#
# # Initialize variables
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # Training loop
# loss_vec = []
# for i in range(1500):
#     rand_index = np.random.choice(len(x_vals), size=batch_size)
#     rand_x = np.transpose([x_vals[rand_index]])
#     rand_y = np.transpose([y_vals[rand_index]])
#     sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
#     temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
#     loss_vec.append(temp_loss[0])
#     if (i + 1) % 300 == 0:
#         print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
#         print('Loss = ' + str(temp_loss))
#         print('\n')
#
# ###
# # Extract regression results
# ###
#
# # Get the optimal coefficients
# [slope] = sess.run(A)
# [y_intercept] = sess.run(b)
#
# # Get best fit line
# best_fit = []
# for i in x_vals:
#     best_fit.append(slope * i + y_intercept)
#
# ###
# # Plot results
# ###
#
# # Plot regression line against data points
# plt.plot(x_vals, y_vals, 'o', label='Data Points')
# plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
# plt.legend(loc='upper left')
# plt.title('Sepal Length vs Pedal Width')
# plt.xlabel('Pedal Width')
# plt.ylabel('Sepal Length')
# plt.show()
#
# # Plot loss over time
# plt.plot(loss_vec, 'k-')
# plt.title(regression_type + ' Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.show()


# 6 用TensorFlow实现弹性网络回归算法

# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from sklearn import datasets
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
# sess = tf.Session()
#
# # Load the data
# # iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# iris = datasets.load_iris()
# x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
# y_vals = np.array([y[0] for y in iris.data])
#
# seed = 13
# np.random.seed(seed)
# tf.set_random_seed(seed)
#
# batch_size = 50
# learning_rate = 0.001
# x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
# y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# A = tf.Variable(tf.random_normal(shape=[3,1]))
# b = tf.Variable(tf.random_normal(shape=[1,1]))
# model_output = tf.add(tf.matmul(x_data, A), b)
#
# # Declare the elastic net loss function
# elastic_param1 = tf.constant(1.)
# elastic_param2 = tf.constant(1.)
# l1_a_loss = tf.reduce_mean(tf.abs(A))
# l2_a_loss = tf.reduce_mean(tf.square(A))
# e1_term = tf.multiply(elastic_param1, l1_a_loss)
# e2_term = tf.multiply(elastic_param2, l2_a_loss)
# loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
#
# # Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.001)
# train_step = my_opt.minimize(loss)
#
#
# # Initialize variables
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # Training loop
# loss_vec = []
# for i in range(1000):
#     rand_index = np.random.choice(len(x_vals), size=batch_size)
#     rand_x = x_vals[rand_index]
#     rand_y = np.transpose([y_vals[rand_index]])
#     sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
#     temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
#     loss_vec.append(temp_loss[0])
#     if (i+1)%250==0:
#         print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
#         print('Loss = ' + str(temp_loss))
#
# # Get the optimal coefficients
# [[sw_coef], [pl_coef], [pw_ceof]] = sess.run(A)
# [y_intercept] = sess.run(b)
#
# # Plot loss over time
# plt.plot(loss_vec, 'k-')
# plt.title('Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.show()


# 7 用TensorFlow实现逻辑回归算法

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from sklearn import datasets
from sklearn.preprocessing import normalize
import os.path
import csv
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()





birth_weight_file = 'birth_weight.csv'
# 如果文件不存在下载
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = birth_data[0].split('\t')
    birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
    with open(birth_weight_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(birth_header)
        writer.writerows(birth_data)
        f.close()
# 读取birth_weight_file到内存
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
     csv_reader = csv.reader(csvfile)
     birth_header = next(csv_reader)
     for row in csv_reader:
         birth_data.append(row)
birth_data = [[float(x) for x in row] for row in birth_data]
"""
数据样式：

LOW	AGE	LWT	RACE	SMOKE	PTL	HT	UI	BWT
1	28	113	1	1	1	0	1	709
1	29	130	0	0	0	0	1	1021
1	34	187	1	1	0	1	0	1135
1	25	105	1	0	1	1	0	1330
"""
y_vals = np.array([x[0] for x in birth_data])
x_vals = np.array([x[1:8] for x in birth_data])








seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)
# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]






# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))







batch_size = 25
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[7,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)






# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)
# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)






# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)






# Training loop
loss_vec = []
train_acc = []
test_acc = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_acc.append(temp_acc_train)
    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_acc.append(temp_acc_test)
    if (i+1)%300==0:
        print('Loss = ' + str(temp_loss))





# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()
# Plot train and test accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
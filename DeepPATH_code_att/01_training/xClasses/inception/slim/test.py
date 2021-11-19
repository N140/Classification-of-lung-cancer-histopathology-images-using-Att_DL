import tensorflow as tf
w = tf.Variable([[1, 13]], dtype=tf.float32)  # w.shape: [1, 2]
score=tf.nn.softmax(w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(score))
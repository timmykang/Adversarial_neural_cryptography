import tensorflow as tf
import numpy as np
import Net
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.logging.set_verbosity(tf.logging.ERROR)

plainTextLength = 16
keyLength = 16
N = plainTextLength / 2
batch = 4096
learningRate = 0.0008
TRAIN_STEP= 200000
iterations = 1
max_eve1_counter = 5000
numbers_of_eve1_reset = 5


def get_random_block(N, batch):
    return 2 * np.random.randint(2, size=(batch, N)) - 1


def train():
	with tf.name_scope('input_variable'):
		plain = tf.placeholder(tf.float32, shape=[None, plainTextLength], name='plainText')
		key = tf.placeholder(tf.float32, shape=[None, keyLength], name='keyText')
		predict_key = tf.placeholder(tf.float32, shape=[None, keyLength], name='predict_keyText')

	Zeros = tf.zeros_like(plain, dtype=tf.float32, name='zeroVector')

	#
	Alice_output, Bob_output, Eve_output, Eve1_output = Net._build_Network(plain, key, predict_key, plainTextLength, keyLength)
	reshape_Bob_output = tf.reshape(Bob_output, shape=[-1, plainTextLength])
	reshape_Eve_output = tf.reshape(Eve_output, shape=[-1, plainTextLength])
	reshape_Eve1_output = tf.reshape(Eve1_output, shape=[-1, plainTextLength])
	# Bob L1 loss
	with tf.name_scope('Bob_loss'):
		Bob_loss = tf.reduce_mean(tf.abs(reshape_Bob_output - plain))
	tf.summary.scalar('Bob_loss_value', Bob_loss)
	# Eve L1 Loss
	with tf.name_scope('Eve_loss'):
		Eve_loss = tf.reduce_mean(tf.abs(reshape_Eve_output - plain))
	tf.summary.scalar('Eve_loss_value', Eve_loss)
	# Alice_Bob Loss
	with tf.name_scope('A_B_loss'):
		Alice_Bob_loss = Bob_loss + (1 - Eve_loss) ** 2
	tf.summary.scalar('Alice_Bob_loss_value', Alice_Bob_loss)

	with tf.name_scope('Eve1_loss'):
		Eve1_loss = tf.reduce_mean(tf.abs(reshape_Eve1_output - plain))
	tf.summary.scalar('Eve1_loss_value', Eve1_loss)

	# error
	boolean_P = tf.greater(plain, Zeros)
	boolean_B = tf.greater_equal(reshape_Bob_output, Zeros)
	boolean_E = tf.greater_equal(reshape_Eve_output, Zeros)
	boolean_E1 = tf.greater_equal(reshape_Eve1_output, Zeros)
	accuracy_B = tf.reduce_mean(tf.cast(tf.equal(boolean_B, boolean_P), dtype=tf.float32))
	accuracy_E = tf.reduce_mean(tf.cast(tf.equal(boolean_E, boolean_P), dtype=tf.float32))
	accuracy_E1 = tf.reduce_mean(tf.cast(tf.equal(boolean_E1, boolean_P), dtype=tf.float32))
	Bob_bits_wrong = plainTextLength - accuracy_B * plainTextLength
	Eve_bits_wrong = plainTextLength - accuracy_E * plainTextLength
	Eve1_bits_wrong = plainTextLength - accuracy_E1 * plainTextLength
	tf.summary.scalar('accuracy_B_value', accuracy_B)
	tf.summary.scalar('accuracy_E_value', accuracy_E)
	tf.summary.scalar('accuracy_E1_value', accuracy_E1)
	tf.summary.scalar('Bob_bits_wrong', Bob_bits_wrong)
	tf.summary.scalar('Eve_bits_wrong', Eve_bits_wrong)
	tf.summary.scalar('Eve1_bits_wrong', Eve1_bits_wrong)

	A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Alice')
	B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Bob')
	AB_vars = A_vars + B_vars
	Eve_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Eve')
	Eve1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Eve1')

	Alice_Bob_optimizer = tf.train.AdamOptimizer(learningRate).minimize(Alice_Bob_loss, var_list=AB_vars)
	Eve_optimizer = tf.train.AdamOptimizer(learningRate).minimize(Eve_loss, var_list=Eve_vars)
	Eve1_optimizer = tf.train.AdamOptimizer(learningRate).minimize(Eve1_loss, var_list=Eve1_vars)

	merged = tf.summary.merge_all()

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		train_writer = tf.summary.FileWriter('E:\\adversarial_neural_cryptography\\cryptanalysis_test\\adver_logs', session.graph)
		if not os.path.exists('adver_logs'):
			os.makedirs('adver_logs')
		f = open('construct_neural_network.txt', 'w')

		for step in range(TRAIN_STEP):
			# train Bob
			tmp_key = get_random_block(keyLength, batch)
			feedDict = {plain: get_random_block(plainTextLength, batch),
						key: tmp_key,
						predict_key: tmp_key.copy()}
			for index in range(iterations):
				_, Bob_error, Bob_accuracy, Bob_wrong_bits = session.run(
					[Alice_Bob_optimizer, Bob_loss, accuracy_B, Bob_bits_wrong], feed_dict=feedDict)
			Bob_accuracy_bits = Bob_accuracy * plainTextLength
			# train Eve
			tmp_key = get_random_block(keyLength, 2 * batch)
			Eve_feedDict = {plain: get_random_block(plainTextLength, 2 * batch),
							key: tmp_key,
							predict_key: tmp_key.copy()}
			for index in range(2 * iterations):
				session.run(Eve_optimizer, feed_dict=Eve_feedDict)
			Eve_error, Eve_accuracy, Eve_wrong_bits = session.run([Eve_loss, accuracy_E, Eve_bits_wrong], feed_dict=Eve_feedDict)
			Eve_accuracy_bits = Eve_accuracy * plainTextLength
			AB_error, summary = session.run([Alice_Bob_loss, merged], feed_dict=feedDict)

			if step % 500 == 0:
				print('Step:', step)
				print('Eve_error:', Eve_error, 'Eve accuracy bits', Eve_accuracy_bits, 'AB_error:', AB_error)
				print('Bob_error:', Bob_error, 'Bob Accuracy Bits:', Bob_accuracy_bits)
				f.write('Step: {}\n'.format(step))
				f.write('Eve_error: {}, Eve accuracy bits: {}, AB_error: {}\n'.format(Eve_error, Eve_accuracy_bits, AB_error))
				f.write('Bob_error: {}, Bob Accuracy bits: {}\n'.format(Bob_error, Bob_accuracy_bits))

			train_writer.add_summary(summary, step)
			if Eve_error > 0.96 and Bob_error < 0.001:
				break
		f.close()

		f = open('Learning_Eve1.txt', 'w')
		flag = 0
		for _ in range(numbers_of_eve1_reset):
			train_writer = tf.summary.FileWriter('E:\\adversarial_neural_cryptography\\cryptanalysis_test\\adver_logs', session.graph)
			session.run(tf.variables_initializer(Eve1_vars))
			for eve1_counter in range(max_eve1_counter):
				tmp_key = get_random_block(keyLength, 2 * batch)
				Eve1_feedDict = {plain: get_random_block(plainTextLength, 2 * batch),
								key: tmp_key,
								predict_key: tmp_key.copy()}
				for index in range(2 * iterations):
					session.run(Eve1_optimizer, feed_dict=Eve1_feedDict)
				Eve1_error, Eve1_accuracy, Eve1_wrong_bits = session.run([Eve1_loss, accuracy_E1, Eve1_bits_wrong], feed_dict=Eve1_feedDict)
				Eve1_accuracy_bits = Eve1_accuracy * plainTextLength
				if eve1_counter % 500 == 0:
					print('Eve1_error:', Eve1_error, 'Eve1 accuracy bits', Eve1_accuracy_bits)
					f.write('Eve1_error: {}, Eve1 accuracy bits: {}\n'.format(Eve1_error, Eve1_accuracy_bits))
				if Eve1_accuracy_bits > 12:
					flag = 1
					break
			train_writer.add_summary(summary, step)	
			if flag == 1:
				break
		f.close()
		
		print('start testing Eve1')
		f = open("testing_Eve1.txt", 'w')
		for _ in range(numbers_of_eve1_reset):
			train_writer = tf.summary.FileWriter('E:\\adversarial_neural_cryptography\\cryptanalysis_test\\adver_logs', session.graph)
			session.run(tf.variables_initializer(Eve1_vars))
			for eve1_counter in range(max_eve1_counter):
				Eve1_feedDict = {plain: get_random_block(plainTextLength, 2 * batch),
								key: get_random_block(keyLength, 2 * batch),
								predict_key: get_random_block(keyLength, 2 * batch)}
				Eve1_error, Eve1_accuracy, Eve1_wrong_bits = session.run([Eve1_loss, accuracy_E1, Eve1_bits_wrong], feed_dict=Eve1_feedDict)
				Eve1_accuracy_bits = Eve1_accuracy * plainTextLength
				if eve1_counter % 500 == 0:
					print('Eve1_error:', Eve1_error, 'Eve1 accuracy bits', Eve1_accuracy_bits)
					f.write('Eve1_error: {}, Eve1 accuracy bits: {}\n'.format(Eve1_error, Eve1_accuracy_bits))
			train_writer.add_summary(summary, step)	
		f.close()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
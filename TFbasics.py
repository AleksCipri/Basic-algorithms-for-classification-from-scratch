##kompajliranje sa sudo python3 Name.py
## htop se kuca u terminalu za prikaz koliko procesor radi
##tf.mul, tf.sub, tf.neg -> tf.multiply, tf.subtract, tf.negative

import tensorflow as tf 

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2)

print(result)

## da bismo mogli da vidimo broj koji je rezultat mnozenja

#sess = tf.Session()
#print(sess.run(result))
#sess.close()

## bolji i kraci nacin pisanja prethodnog 
## da ne moramo da kucamo open i close za session
with tf.Session() as sess:
	# kada dodelimo promenljivoj output, 
	# mozemo da joj pristupimo i nakon zatvaranja sesije
	output = sess.run(result)
	print(output)
	#print(sess.run(result))
print(output)
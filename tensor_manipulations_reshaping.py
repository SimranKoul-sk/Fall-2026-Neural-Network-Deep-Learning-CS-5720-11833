import tensorflow as tf

#Make random values repeatable at every run
tf.random.set_seed(42)

#Create the original 2D tensor
#4 rows x 6 columns = 24 elements.
original_tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.int32)

#tf.rank = the number of dimensions
#tf.shape = dimension sizes
print("Original tensor:")
print(original_tensor.numpy())
print("Rank:", tf.rank(original_tensor).numpy())
print("Shape:", tf.shape(original_tensor).numpy())

#Reshaping w/o changing data
#4 * 6 = 2 * 3 * 4 = 24 elements
reshaped_tensor = tf.reshape(original_tensor, shape=(2, 3, 4))

print("\nReshaped tensor (2, 3, 4):")
print(reshaped_tensor.numpy())
print("Rank:", tf.rank(reshaped_tensor).numpy())
print("Shape:", tf.shape(reshaped_tensor).numpy())

#Reorder the first 2 axis
#The last axis (size 4) remains the same
transposed_tensor = tf.transpose(reshaped_tensor, perm=(1, 0, 2))

print("\nTransposed tensor (3, 2, 4):")
print(transposed_tensor.numpy())
print("Rank:", tf.rank(transposed_tensor).numpy())
print("Shape:", tf.shape(transposed_tensor).numpy())

#(1, 4) tensor gives 4 values for final dimension
small_tensor = tf.constant([[10, 20, 30, 40]], dtype=tf.int32)

#TensorFlow gives (1, 4) to (3, 2, 4) during addition
#The missing dimension is treated as 1 and repeated to 3
#The dimension of size 1 is repeated to size 2
#The final dimension already has matching size 4
result_tensor = transposed_tensor + small_tensor

print("\nSmaller tensor (1, 4):")
print(small_tensor.numpy())
print("Shape:", tf.shape(small_tensor).numpy())
print("\nResult after broadcasting and addition (3, 2, 4):")
print(result_tensor.numpy())
print("Rank:", tf.rank(result_tensor).numpy())
print("Shape:", tf.shape(result_tensor).numpy())

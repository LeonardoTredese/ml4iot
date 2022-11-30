import tensorflow as tf


train_files_ds = tf.data.Dataset.list_files('../msc-data/msc-train')
print(type(train_files_ds))
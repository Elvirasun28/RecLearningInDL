import tensorflow as tf
'''
根据DeepFM的思想，我们需要将每一个field的特征转换为定长的embedding，
即使有多个取值，也是要变换成定长的embedding。

那么，一种思路来了，比如一个用户喜欢两个球队，这个field的特征可能是[1,1,0,0,0,0,0.....0]，
那么我们使用两次embedding lookup，再取个平均不就好了嘛。

嗯，这的确也许可能是一种思路吧，在tensorflow中，其实有一个函数能够实现我们上述的思路，
那就是tf.nn.embedding_lookup_sparse
'''

''' 
loading dataset
假设我们有三条数据，每条数据代表一个user所喜欢的nba球员，比如有登哥，炮哥，
杜老四，慕斯等等：
'''
csv = [
  "1,harden|james|curry",
  "2,wrestbrook|harden|durant",
  "3,|paul|towns",
]
TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"] ## all nba superstar numbers

''' 
Data Processing 
这里我们需要一个得到一个SparseTensor，即多为稀疏矩阵的一种表示方式，我们只记录非0值所在的位置和值。

比如说，下面就是我们对上面数据处理过后的一个SparseTensor，indices是数组中非0元素的下标，values跟indices一一对应，
表示该下标位置的值，最后一个表示的是数组的大小。
'''
def sparse_from_csv(csv):
    ids, post_tags_str = tf.decode_csv(csv,[[-1],[""]])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping = TAG_SET, default_value = -1
    ) ## build search table
    split_tags = tf.string_split(post_tags_str,"|")
    return tf.SparseTensor(
        indices=split_tags.indices,
        values=table.lookup(split_tags.values),
        dense_shape=split_tags.dense_shape
    )

''' define embedding params '''
TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))

''' embedding values '''
tags = sparse_from_csv(csv)
embedding_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)


with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print(s.run([embedding_tags]))
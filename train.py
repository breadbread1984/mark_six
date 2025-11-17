#!/usr/bin/python3

from absl import flags, app
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to input xlsx')

def main(unused_argv):
  df = pd.read_excel(FLAGS.input)
  samples = df[['中獎號碼 1','2','3','4','5','6','特別號碼']].to_numpy() # samples.shape = (num_samples, 7)
  num_balls = 49
  sample_size = 7

  # 设置参数先验，Dirichlet分布保证theta在概率单纯形上
  alpha = tf.ones(num_balls, dtype=tf.float32)  # 可调整先验强度

  def log_likelihood(theta):
    # theta是shape=(49,)的概率分布向量
    # 计算所有采样观测数据的似然log概率总和
    log_prob = 0.
    for draw in samples:
      draw = tf.convert_to_tensor(draw)
      # draw.shape = (7, )
      # draw是7个数字，索引从1~49转为0~48
      indices = draw - 1
        
      # 模拟“无放回”加权采样的近似似然计算：
      # 这里近似用log prod theta[indices], 理论上无放回要复杂计算
      log_prob += tf.reduce_sum(tf.math.log(tf.gather(theta, indices)))
    return log_prob

  def target_log_prob_fn(theta_unconstrained):
    # 先通过softmax变换将无约束变量映射到概率空间
    theta = tf.nn.softmax(theta_unconstrained)
    log_prior = tfd.Dirichlet(alpha).log_prob(theta)
    log_lik = log_likelihood(theta)
    return log_prior + log_lik

  num_results = 1000
  num_burnin_steps = 500

  initial_state = tf.zeros(num_balls)

  kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=0.01,
    num_leapfrog_steps=3
  )

  transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=kernel,
    bijector=tfb.SoftmaxCentered()
 )

  @tf.function
  def run_chain():
    samples, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=tf.zeros(num_balls),
      kernel=transformed_kernel,
      num_burnin_steps=num_burnin_steps
    )
    return samples, kernel_results

  samples, results = run_chain()
  np.save('params.npy', samples.numpy())

if __name__ == "__main__":
  add_options()
  app.run(main)

#ifndef HYBRID_BNN_COMMON_H
#define HYBRID_BNN_COMMON_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

inline double hybrid_bnn_sigmoid(double x)
{
  if (x >= 0.0) {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }

  const double z = std::exp(x);
  return z / (1.0 + z);
}

template <std::size_t FEATURES>
class BayesianNNPredictor
{
public:
  using feature_array = std::array<double, FEATURES>;

  struct prediction {
    bool taken = false;
    double mean_probability = 0.5;
    double uncertainty = 1.0;
  };

private:
  struct training_example {
    feature_array features = {};
    double label = 0.0;
  };

  struct forward_cache {
    std::vector<double> input;
    std::vector<double> z1;
    std::vector<double> a1;
    std::vector<double> d1;
    std::vector<double> mask1;
    std::vector<double> z2;
    std::vector<double> a2;
    std::vector<double> d2;
    std::vector<double> mask2;
    double logit = 0.0;
  };

  class LinearLayer
  {
    int input_size_ = 0;
    int output_size_ = 0;
    std::vector<double> weights_;
    std::vector<double> biases_;
    std::vector<double> grad_weights_;
    std::vector<double> grad_biases_;
    std::vector<double> adam_m_weights_;
    std::vector<double> adam_v_weights_;
    std::vector<double> adam_m_biases_;
    std::vector<double> adam_v_biases_;

  public:
    LinearLayer() = default;

    LinearLayer(int input_size, int output_size, std::mt19937& rng)
        : input_size_(input_size),
          output_size_(output_size),
          weights_(static_cast<std::size_t>(input_size) * static_cast<std::size_t>(output_size)),
          biases_(static_cast<std::size_t>(output_size)),
          grad_weights_(weights_.size(), 0.0),
          grad_biases_(biases_.size(), 0.0),
          adam_m_weights_(weights_.size(), 0.0),
          adam_v_weights_(weights_.size(), 0.0),
          adam_m_biases_(biases_.size(), 0.0),
          adam_v_biases_(biases_.size(), 0.0)
    {
      const double bound = 1.0 / std::sqrt(std::max(1, input_size_));
      std::uniform_real_distribution<double> dist(-bound, bound);

      for (double& weight : weights_)
        weight = dist(rng);
      for (double& bias : biases_)
        bias = dist(rng);
    }

    void zero_grad()
    {
      std::fill(std::begin(grad_weights_), std::end(grad_weights_), 0.0);
      std::fill(std::begin(grad_biases_), std::end(grad_biases_), 0.0);
    }

    void forward(const std::vector<double>& input, std::vector<double>& output) const
    {
      output.assign(static_cast<std::size_t>(output_size_), 0.0);

      for (int out = 0; out < output_size_; ++out) {
        double sum = biases_[static_cast<std::size_t>(out)];
        const std::size_t row_offset = static_cast<std::size_t>(out) * static_cast<std::size_t>(input_size_);
        for (int in = 0; in < input_size_; ++in)
          sum += weights_[row_offset + static_cast<std::size_t>(in)] * input[static_cast<std::size_t>(in)];

        output[static_cast<std::size_t>(out)] = sum;
      }
    }

    void backward(const std::vector<double>& input, const std::vector<double>& grad_output, std::vector<double>& grad_input)
    {
      grad_input.assign(static_cast<std::size_t>(input_size_), 0.0);

      for (int out = 0; out < output_size_; ++out) {
        const double grad = grad_output[static_cast<std::size_t>(out)];
        grad_biases_[static_cast<std::size_t>(out)] += grad;

        const std::size_t row_offset = static_cast<std::size_t>(out) * static_cast<std::size_t>(input_size_);
        for (int in = 0; in < input_size_; ++in) {
          const std::size_t idx = row_offset + static_cast<std::size_t>(in);
          grad_weights_[idx] += grad * input[static_cast<std::size_t>(in)];
          grad_input[static_cast<std::size_t>(in)] += weights_[idx] * grad;
        }
      }
    }

    void adam_step(double learning_rate, int step)
    {
      constexpr double beta1 = 0.9;
      constexpr double beta2 = 0.999;
      constexpr double epsilon = 1e-8;

      const double bias_correction1 = 1.0 - std::pow(beta1, step);
      const double bias_correction2 = 1.0 - std::pow(beta2, step);

      for (std::size_t i = 0; i < weights_.size(); ++i) {
        adam_m_weights_[i] = beta1 * adam_m_weights_[i] + (1.0 - beta1) * grad_weights_[i];
        adam_v_weights_[i] = beta2 * adam_v_weights_[i] + (1.0 - beta2) * grad_weights_[i] * grad_weights_[i];
        const double m_hat = adam_m_weights_[i] / bias_correction1;
        const double v_hat = adam_v_weights_[i] / bias_correction2;
        weights_[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      for (std::size_t i = 0; i < biases_.size(); ++i) {
        adam_m_biases_[i] = beta1 * adam_m_biases_[i] + (1.0 - beta1) * grad_biases_[i];
        adam_v_biases_[i] = beta2 * adam_v_biases_[i] + (1.0 - beta2) * grad_biases_[i] * grad_biases_[i];
        const double m_hat = adam_m_biases_[i] / bias_correction1;
        const double v_hat = adam_v_biases_[i] / bias_correction2;
        biases_[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
      }
    }
  };

  double dropout_probability_;
  double keep_probability_;
  double learning_rate_;
  double uncertainty_threshold_;
  std::size_t mc_samples_;
  std::size_t batch_size_;
  std::size_t replay_capacity_;
  std::size_t train_interval_;
  std::size_t warmup_branches_;
  std::size_t observations_ = 0;
  int optimizer_step_ = 0;
  mutable std::mt19937 rng_;
  LinearLayer layer1_;
  LinearLayer layer2_;
  LinearLayer layer3_;
  std::deque<training_example> replay_buffer_;

  void apply_dropout(const std::vector<double>& input, std::vector<double>& output, std::vector<double>& mask, bool dropout_active)
  {
    output.resize(input.size());
    mask.resize(input.size());

    if (!dropout_active || dropout_probability_ <= 0.0) {
      std::fill(std::begin(mask), std::end(mask), 1.0);
      output = input;
      return;
    }

    std::bernoulli_distribution keep_dist(keep_probability_);
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (keep_dist(rng_)) {
        mask[i] = 1.0 / keep_probability_;
        output[i] = input[i] * mask[i];
      } else {
        mask[i] = 0.0;
        output[i] = 0.0;
      }
    }
  }

  void forward_internal(const feature_array& features, forward_cache& cache, bool dropout_active)
  {
    cache.input.assign(std::begin(features), std::end(features));

    layer1_.forward(cache.input, cache.z1);
    cache.a1 = cache.z1;
    for (double& value : cache.a1)
      value = std::max(0.0, value);
    apply_dropout(cache.a1, cache.d1, cache.mask1, dropout_active);

    layer2_.forward(cache.d1, cache.z2);
    cache.a2 = cache.z2;
    for (double& value : cache.a2)
      value = std::max(0.0, value);
    apply_dropout(cache.a2, cache.d2, cache.mask2, dropout_active);

    std::vector<double> output;
    layer3_.forward(cache.d2, output);
    cache.logit = output.front();
  }

  void backward_sample(const forward_cache& cache, double label, double batch_scale)
  {
    const double grad_logit = (hybrid_bnn_sigmoid(cache.logit) - label) / batch_scale;

    std::vector<double> grad_layer3(1, grad_logit);
    std::vector<double> grad_d2;
    layer3_.backward(cache.d2, grad_layer3, grad_d2);

    std::vector<double> grad_a2 = grad_d2;
    for (std::size_t i = 0; i < grad_a2.size(); ++i)
      grad_a2[i] *= cache.mask2[i];

    std::vector<double> grad_z2 = grad_a2;
    for (std::size_t i = 0; i < grad_z2.size(); ++i) {
      if (cache.z2[i] <= 0.0)
        grad_z2[i] = 0.0;
    }

    std::vector<double> grad_d1;
    layer2_.backward(cache.d1, grad_z2, grad_d1);

    std::vector<double> grad_a1 = grad_d1;
    for (std::size_t i = 0; i < grad_a1.size(); ++i)
      grad_a1[i] *= cache.mask1[i];

    std::vector<double> grad_z1 = grad_a1;
    for (std::size_t i = 0; i < grad_z1.size(); ++i) {
      if (cache.z1[i] <= 0.0)
        grad_z1[i] = 0.0;
    }

    std::vector<double> grad_input;
    layer1_.backward(cache.input, grad_z1, grad_input);
  }

  void train_from_replay()
  {
    const std::size_t effective_batch = std::min(batch_size_, replay_buffer_.size());
    if (effective_batch == 0)
      return;

    layer1_.zero_grad();
    layer2_.zero_grad();
    layer3_.zero_grad();

    std::uniform_int_distribution<std::size_t> sample_dist(0, replay_buffer_.size() - 1);
    for (std::size_t sample = 0; sample < effective_batch; ++sample) {
      const auto& example = replay_buffer_[sample_dist(rng_)];
      forward_cache cache;
      forward_internal(example.features, cache, true);
      backward_sample(cache, example.label, static_cast<double>(effective_batch));
    }

    ++optimizer_step_;
    layer1_.adam_step(learning_rate_, optimizer_step_);
    layer2_.adam_step(learning_rate_, optimizer_step_);
    layer3_.adam_step(learning_rate_, optimizer_step_);
  }

public:
  BayesianNNPredictor(int hidden1, int hidden2, double dropout_probability, double learning_rate, std::size_t mc_samples,
                      std::size_t batch_size, std::size_t replay_capacity, std::size_t train_interval,
                      std::size_t warmup_branches, double uncertainty_threshold, std::uint32_t seed)
      : dropout_probability_(std::clamp(dropout_probability, 0.0, 0.75)),
        keep_probability_(1.0 - dropout_probability_),
        learning_rate_(learning_rate),
        uncertainty_threshold_(uncertainty_threshold),
        mc_samples_(std::max<std::size_t>(1, mc_samples)),
        batch_size_(std::max<std::size_t>(1, batch_size)),
        replay_capacity_(std::max<std::size_t>(batch_size_, replay_capacity)),
        train_interval_(std::max<std::size_t>(1, train_interval)),
        warmup_branches_(warmup_branches),
        rng_(seed),
        layer1_(static_cast<int>(FEATURES), hidden1, rng_),
        layer2_(hidden1, hidden2, rng_),
        layer3_(hidden2, 1, rng_)
  {
  }

  prediction predict(const feature_array& features)
  {
    prediction result;
    if (observations_ < warmup_branches_ || replay_buffer_.size() < batch_size_)
      return result;

    double probability_sum = 0.0;
    double probability_sq_sum = 0.0;

    for (std::size_t sample = 0; sample < mc_samples_; ++sample) {
      forward_cache cache;
      forward_internal(features, cache, true);
      const double probability = hybrid_bnn_sigmoid(cache.logit);
      probability_sum += probability;
      probability_sq_sum += probability * probability;
    }

    result.mean_probability = probability_sum / static_cast<double>(mc_samples_);
    result.uncertainty = std::max(0.0, probability_sq_sum / static_cast<double>(mc_samples_) - result.mean_probability * result.mean_probability);
    result.taken = result.mean_probability >= 0.5;
    return result;
  }

  void observe(const feature_array& features, bool taken)
  {
    if (replay_buffer_.size() >= replay_capacity_)
      replay_buffer_.pop_front();

    replay_buffer_.push_back(training_example{features, taken ? 1.0 : 0.0});
    ++observations_;

    if (replay_buffer_.size() >= batch_size_ && observations_ >= warmup_branches_ && observations_ % train_interval_ == 0)
      train_from_replay();
  }

  bool is_confident(const prediction& result) const { return result.uncertainty <= uncertainty_threshold_; }
};

#endif

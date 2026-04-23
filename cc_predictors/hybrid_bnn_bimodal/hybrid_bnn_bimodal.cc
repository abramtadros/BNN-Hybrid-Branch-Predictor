#include "hybrid_bnn_bimodal.h"

void hybrid_bnn_bimodal::push_history(history_type& history, bool taken)
{
  history <<= 1;
  history.set(0, taken);
}

bool hybrid_bnn_bimodal::is_weak_counter(int counter)
{
  return counter == 1 || counter == 2;
}

std::size_t hybrid_bnn_bimodal::table_index(champsim::address ip)
{
  return ip.to<unsigned long>() % BIMODAL_PRIME;
}

auto hybrid_bnn_bimodal::encode_features(champsim::address ip, history_type history, const base_prediction& base) -> bnn_type::feature_array
{
  bnn_type::feature_array features = {};
  const auto shifted_ip = ip.to<uint64_t>() >> 2U;

  for (std::size_t bit = 0; bit < PC_FEATURE_BITS; ++bit)
    features[bit] = ((shifted_ip >> bit) & 1U) != 0U ? 1.0 : -1.0;

  for (std::size_t bit = 0; bit < GLOBAL_HISTORY_LENGTH; ++bit)
    features[PC_FEATURE_BITS + bit] = history[bit] ? 1.0 : -1.0;

  constexpr double maximum_counter = static_cast<double>((1U << COUNTER_BITS) - 1U);
  features[PC_FEATURE_BITS + GLOBAL_HISTORY_LENGTH] = (2.0 * static_cast<double>(base.counter) / maximum_counter) - 1.0;
  features[PC_FEATURE_BITS + GLOBAL_HISTORY_LENGTH + 1] = base.taken ? 1.0 : -1.0;
  features[PC_FEATURE_BITS + GLOBAL_HISTORY_LENGTH + 2] = base.weak ? 1.0 : -1.0;

  return features;
}

auto hybrid_bnn_bimodal::base_predict(champsim::address ip) const -> base_prediction
{
  const auto index = table_index(ip);
  const auto value = bimodal_table[index];
  const int counter = static_cast<int>(value.value());
  return base_prediction{value.value() > (value.maximum / 2), counter, index, is_weak_counter(counter)};
}

void hybrid_bnn_bimodal::update_base(const base_prediction& prediction, bool taken)
{
  bimodal_table[prediction.index] += taken ? 1 : -1;
}

bool hybrid_bnn_bimodal::predict_branch(champsim::address ip)
{
  const auto base = base_predict(ip);
  const auto features = encode_features(ip, speculative_history, base);
  const auto bnn = bnn_predictor.predict(features);

  // Use BNN when base is weak and BNN is confident
  const bool use_bnn = base.weak && bnn_predictor.is_confident(bnn);
  const bool final_prediction = use_bnn ? bnn.taken : base.taken;

  state_buf.push_back(prediction_state{ip, final_prediction, base, features, bnn});
  if (std::size(state_buf) > MAX_INFLIGHT_BRANCHES)
    state_buf.pop_front();

  push_history(speculative_history, final_prediction);
  return final_prediction;
}

void hybrid_bnn_bimodal::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  auto state = std::find_if(std::begin(state_buf), std::end(state_buf), [ip](const auto& entry) { return entry.ip == ip; });
  if (state == std::end(state_buf)) {
    const auto base = base_predict(ip);
    const auto features = encode_features(ip, committed_history, base);
    update_base(base, taken);
    bnn_predictor.observe(features, taken);
    push_history(committed_history, taken);
    speculative_history = committed_history;
    return;
  }

  const auto saved = *state;
  state_buf.erase(state);

  update_base(saved.base, taken);
  bnn_predictor.observe(saved.features, taken);

  push_history(committed_history, taken);
  if (saved.final_prediction != taken)
    speculative_history = committed_history;
}

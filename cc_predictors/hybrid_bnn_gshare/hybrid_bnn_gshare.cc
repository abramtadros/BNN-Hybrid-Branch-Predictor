#include "hybrid_bnn_gshare.h"

std::size_t hybrid_bnn_gshare::gs_table_hash(champsim::address ip, history_type history)
{
  constexpr champsim::data::bits LOG2_HISTORY_TABLE_SIZE{champsim::lg2(GS_HISTORY_TABLE_SIZE)};
  constexpr champsim::data::bits LENGTH{GLOBAL_HISTORY_LENGTH};

  std::size_t hash = history.to_ullong();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE, champsim::data::bits{}>().to<std::size_t>();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + LENGTH, LENGTH>().to<std::size_t>();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + 2 * LENGTH, 2 * LENGTH>().to<std::size_t>();

  return hash % GS_HISTORY_TABLE_SIZE;
}

void hybrid_bnn_gshare::push_history(history_type& history, bool taken)
{
  history <<= 1;
  history.set(0, taken);
}

bool hybrid_bnn_gshare::is_weak_counter(int counter)
{
  return counter == 1 || counter == 2;
}

auto hybrid_bnn_gshare::encode_features(champsim::address ip, history_type history, const base_prediction& base) -> bnn_type::feature_array
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

auto hybrid_bnn_gshare::base_predict(champsim::address ip, history_type history) const -> base_prediction
{
  const auto index = gs_table_hash(ip, history);
  const auto value = gs_history_table[index];
  const int counter = static_cast<int>(value.value());
  return base_prediction{value.value() > (value.maximum / 2), counter, index, is_weak_counter(counter)};
}

void hybrid_bnn_gshare::update_base(const base_prediction& prediction, bool taken)
{
  gs_history_table[prediction.index] += taken ? 1 : -1;
}

bool hybrid_bnn_gshare::predict_branch(champsim::address ip)
{
  const auto base = base_predict(ip, speculative_history);
  const auto features = encode_features(ip, speculative_history, base);
  const auto bnn = bnn_predictor.predict(features);

  // Use BNN if base is weak AND BNN is confident
  const bool use_bnn = base.weak && bnn_predictor.is_confident(bnn);
  const bool final_prediction = use_bnn ? bnn.taken : base.taken;

  state_buf.push_back(prediction_state{ip, final_prediction, base, features, bnn});
  if (std::size(state_buf) > MAX_INFLIGHT_BRANCHES)
    state_buf.pop_front();

  push_history(speculative_history, final_prediction);
  return final_prediction;
}

void hybrid_bnn_gshare::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  auto state = std::find_if(std::begin(state_buf), std::end(state_buf), [ip](const auto& entry) { return entry.ip == ip; });
  if (state == std::end(state_buf)) {
    const auto base = base_predict(ip, committed_history);
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


#ifndef BRANCH_HYBRID_BNN_GSHARE_SIMPLE_H
#define BRANCH_HYBRID_BNN_GSHARE_SIMPLE_H

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <deque>

#include "address.h"
#include "hybrid_bnn_common.h"
#include "modules.h"
#include "msl/fwcounter.h"

class hybrid_bnn_gshare_simple : public champsim::modules::branch_predictor
{
  static constexpr std::size_t GLOBAL_HISTORY_LENGTH = 14;
  static constexpr std::size_t COUNTER_BITS = 2;
  static constexpr std::size_t GS_HISTORY_TABLE_SIZE = 16384;
  static constexpr std::size_t PC_FEATURE_BITS = 16;
  static constexpr std::size_t AUXILIARY_FEATURES = 3;
  static constexpr std::size_t BNN_INPUT_SIZE = PC_FEATURE_BITS + GLOBAL_HISTORY_LENGTH + AUXILIARY_FEATURES;
  static constexpr std::size_t MAX_INFLIGHT_BRANCHES = 256;

  using history_type = std::bitset<GLOBAL_HISTORY_LENGTH>;
  using counter_type = champsim::msl::fwcounter<COUNTER_BITS>;
  using bnn_type = BayesianNNPredictor<BNN_INPUT_SIZE>;

  struct base_prediction {
    bool taken = false;
    int counter = 0;
    std::size_t index = 0;
    bool weak = false;
  };

  struct prediction_state {
    champsim::address ip{};
    bool final_prediction = false;
    base_prediction base = {};
    bnn_type::feature_array features = {};
    bnn_type::prediction bnn = {};
  };

  history_type speculative_history = {};
  history_type committed_history = {};
  std::array<counter_type, GS_HISTORY_TABLE_SIZE> gs_history_table = {};
  std::deque<prediction_state> state_buf = {};
  bnn_type bnn_predictor{24, 12, 0.20, 1e-3, 8, 32, 2048, 8, 128, 0.02, 7};

  static std::size_t gs_table_hash(champsim::address ip, history_type history);
  static void push_history(history_type& history, bool taken);
  static bool is_weak_counter(int counter);
  static bnn_type::feature_array encode_features(champsim::address ip, history_type history, const base_prediction& base);

  base_prediction base_predict(champsim::address ip, history_type history) const;
  void update_base(const base_prediction& prediction, bool taken);

public:
  using branch_predictor::branch_predictor;

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

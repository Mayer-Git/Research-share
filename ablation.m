%% ============================================================
%  Ablation Tables 6, 7, and 8 on Segment 2
%
%  Purpose:
%    Generate the exact metrics required by:
%      Table 6. Core mechanism-attribution ablation study
%      Table 7. Dynamic weighting behavior
%      Table 8. Effect of negotiation rounds on coordination and latency
%
%  Required files:
%    xselect.xlsx
%    y1select.xlsx
%    y2select.xlsx
%    y3select.xlsx
%    selected_segments.mat
%
%  Output:
%    ablation_tables_6_7_8_segment2.mat
%
%% ============================================================

clear; clc; close all;

%% ------------------------------------------------------------
%  1. Basic settings
%% ------------------------------------------------------------

rng(2026);

data_path = 'C:/Users/10123/Desktop/优化/实验/重构实验';

file_x  = fullfile(data_path, 'xselect.xlsx');
file_y1 = fullfile(data_path, 'y1select.xlsx');
file_y2 = fullfile(data_path, 'y2select.xlsx');
file_y3 = fullfile(data_path, 'y3select.xlsx');

segment_file = 'selected_segments.mat';
result_file  = 'ablation_tables_6_7_8_segment2.mat';

target_segment_id = 2;

params = struct();

params.train_ratio = 0.7;

params.num_agents = 3;
params.num_objectives = 3;

params.num_candidates = 4000;
params.num_epochs = 150;
params.batch_size = 64;

params.actor_learning_rate = 0.08;
params.exploration_noise_init = 0.12;
params.exploration_noise_decay = 0.985;

params.authority_temperature = 0.35;
params.authority_min_weight = 0.08;
params.authority_smoothing_alpha = 0.35;
params.utility_smoothing_lambda = 0.5;

% IMPORTANT:
% Use K_n = 4 as the default full RGN-MARL setting,
% because Table 8 compares K_n = 1,2,3,4.
params.max_negotiation_rounds_default = 4;

params.ngaf_step_size = 0.12;
params.consensus_lambda = 0.35;
params.ngaf_converge_tol = 1e-4;

params.knn_k = 5;
params.ood_gamma = 2.0;
params.ood_threshold_quantile = 0.85;

params.beta_consensus = 0.15;
params.gamma_ood_reward = 0.5;

params.num_eval_rollouts = 120;

params.collapse_threshold = 0.85;

%% ------------------------------------------------------------
%  2. Load data
%% ------------------------------------------------------------

fprintf('Loading data...\n');

X  = readmatrix(file_x);
Y1 = readmatrix(file_y1);
Y2 = readmatrix(file_y2);
Y3 = readmatrix(file_y3);

Y1 = Y1(:);
Y2 = Y2(:);
Y3 = Y3(:);

valid_idx = all(~isnan(X), 2) & ~isnan(Y1) & ~isnan(Y2) & ~isnan(Y3);

X  = X(valid_idx, :);
Y1 = Y1(valid_idx);
Y2 = Y2(valid_idx);
Y3 = Y3(valid_idx);

Y = [Y1, Y2, Y3];

num_samples = size(X, 1);
num_features = size(X, 2);

params.num_features = num_features;

fprintf('Valid samples: %d\n', num_samples);
fprintf('Input variables: %d\n', num_features);

%% ------------------------------------------------------------
%  3. Load selected Segment 2
%% ------------------------------------------------------------

if ~exist(segment_file, 'file')
    error('selected_segments.mat not found. Please run the segment selection script first.');
end

load(segment_file, 'segments', 'segment_len', 'num_segments');

if target_segment_id > num_segments
    error('target_segment_id exceeds available selected segments.');
end

idx_ccsd = segments{target_segment_id};

fprintf('\nUsing Segment %d for ablation experiments.\n', target_segment_id);
fprintf('CCSD-supported segment range: %d -- %d\n', idx_ccsd(1), idx_ccsd(end));

% Raw-style expanded segment for RGN-MARL w/o CCSD.
% It is centered on Segment 2 but expanded to mimic no steady-state filtering.
raw_expand = segment_len;

raw_start = max(1, idx_ccsd(1) - raw_expand);
raw_end   = min(num_samples, idx_ccsd(end) + raw_expand);

idx_raw = raw_start:raw_end;

fprintf('Raw-style expanded range for w/o CCSD: %d -- %d\n', idx_raw(1), idx_raw(end));

%% ------------------------------------------------------------
%  4. Prepare CCSD-supported data source
%% ------------------------------------------------------------

fprintf('\nPreparing CCSD-supported data source...\n');

data_ccsd = prepare_data_source( ...
    X, Y, idx_ccsd, params, 'CCSD-supported Segment 2');

%% ------------------------------------------------------------
%  5. Prepare raw-style data source for w/o CCSD
%% ------------------------------------------------------------

fprintf('\nPreparing raw-style data source for w/o CCSD...\n');

data_raw = prepare_data_source( ...
    X, Y, idx_raw, params, 'Raw-style expanded Segment 2');

%% ------------------------------------------------------------
%  6. Define Table 6 core ablation variants
%% ------------------------------------------------------------

core_variants = struct();

core_variants(1).name = 'RGN-MARL w/o CCSD';
core_variants(1).data_mode = 'raw';
core_variants(1).use_confidence_replay = false;
core_variants(1).weight_mode = 'deficiency';
core_variants(1).max_rounds = params.max_negotiation_rounds_default;

core_variants(2).name = 'Hard CCSD only';
core_variants(2).data_mode = 'ccsd';
core_variants(2).use_confidence_replay = false;
core_variants(2).weight_mode = 'deficiency';
core_variants(2).max_rounds = params.max_negotiation_rounds_default;

core_variants(3).name = 'Fixed weights';
core_variants(3).data_mode = 'ccsd';
core_variants(3).use_confidence_replay = true;
core_variants(3).weight_mode = 'fixed';
core_variants(3).max_rounds = params.max_negotiation_rounds_default;

core_variants(4).name = 'Utility-proportional weights';
core_variants(4).data_mode = 'ccsd';
core_variants(4).use_confidence_replay = true;
core_variants(4).weight_mode = 'utility';
core_variants(4).max_rounds = params.max_negotiation_rounds_default;

core_variants(5).name = 'K_n=1 one-shot fusion';
core_variants(5).data_mode = 'ccsd';
core_variants(5).use_confidence_replay = true;
core_variants(5).weight_mode = 'deficiency';
core_variants(5).max_rounds = 1;

core_variants(6).name = 'RGN-MARL full';
core_variants(6).data_mode = 'ccsd';
core_variants(6).use_confidence_replay = true;
core_variants(6).weight_mode = 'deficiency';
core_variants(6).max_rounds = params.max_negotiation_rounds_default;

num_core_variants = numel(core_variants);

%% ------------------------------------------------------------
%  7. Run Table 6 core ablation variants
%% ------------------------------------------------------------

%% ------------------------------------------------------------
%  7. Run Table 6 core ablation variants
%% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('Running Table 6 core mechanism-attribution ablations...\n');
fprintf('============================================================\n');

core_results_cell = cell(num_core_variants, 1);

for v = 1:num_core_variants

    rng(2026);

    cfg = core_variants(v);

    if strcmp(cfg.data_mode, 'raw')
        data_source = data_raw;
    elseif strcmp(cfg.data_mode, 'ccsd')
        data_source = data_ccsd;
    else
        error('Unknown data_mode: %s', cfg.data_mode);
    end

    fprintf('\n------------------------------------------------------------\n');
    fprintf('Running core variant %d/%d: %s\n', ...
        v, num_core_variants, cfg.name);
    fprintf('------------------------------------------------------------\n');

    tmp_result = run_single_variant(cfg, data_source, params);

    % 强制保存变体编号和名称，方便后续检查
    tmp_result.variant_id = v;
    tmp_result.variant_name = cfg.name;

    core_results_cell{v} = tmp_result;

end

% 检查是否所有核心消融实验都成功保存
missing_idx = find(cellfun(@isempty, core_results_cell));

if ~isempty(missing_idx)
    disp('Missing core variant indices:');
    disp(missing_idx);
    error('Some core ablation variants were not successfully stored.');
end

% 循环结束后，再统一拼接结构体数组
core_results = concatenate_structs_safely(core_results_cell);

fprintf('\nCore variants successfully stored: %d / %d\n', ...
    numel(core_results), num_core_variants);

for v = 1:numel(core_results)
    fprintf('  %d. %s\n', v, core_results(v).variant_name);
end


%% ------------------------------------------------------------
%  8. Compute IGD and Spread for Table 6
%% ------------------------------------------------------------

fprintf('\nComputing IGD and Spread for Table 6...\n');

all_eval_Y_core = [];

for v = 1:num_core_variants
    all_eval_Y_core = [all_eval_Y_core; core_results(v).eval_Y];
end

C_all_core = objective_to_cost(all_eval_Y_core);
nd_idx_core = get_nondominated_indices(C_all_core);
Y_ref_core = all_eval_Y_core(nd_idx_core, :);

for v = 1:num_core_variants

    core_results(v).IGD = compute_igd(Y_ref_core, core_results(v).eval_Y);
    core_results(v).Spread = compute_spread(core_results(v).eval_Y);

    core_results(v).TD_error_var = compute_td_error_variance( ...
        core_results(v).global_reward_history);

    [avgH, stdH, collapseR] = compute_weight_entropy_metrics( ...
        core_results(v).authority_history, params.collapse_threshold);

    core_results(v).weight_entropy = avgH;
    core_results(v).entropy_std = stdH;
    core_results(v).collapse_ratio = collapseR;

    core_results(v).stability = compute_training_stability( ...
        core_results(v).global_reward_history);

end

%% ------------------------------------------------------------
%  9. Build Table 6
%% ------------------------------------------------------------

Variant = strings(num_core_variants, 1);
IGD = zeros(num_core_variants, 1);
Spread = zeros(num_core_variants, 1);
TD_error_Var = zeros(num_core_variants, 1);
Weight_Entropy = strings(num_core_variants, 1);
Stability = zeros(num_core_variants, 1);

for v = 1:num_core_variants

    Variant(v) = string(core_results(v).variant_name);
    IGD(v) = core_results(v).IGD;
    Spread(v) = core_results(v).Spread;
    TD_error_Var(v) = core_results(v).TD_error_var;
    Stability(v) = core_results(v).stability;

    % In your Table 6, the first two rows use '-' for Weight Entropy.
    if strcmp(core_results(v).variant_name, 'RGN-MARL w/o CCSD') || ...
       strcmp(core_results(v).variant_name, 'Hard CCSD only')
        Weight_Entropy(v) = "-";
    else
        Weight_Entropy(v) = string(sprintf('%.4f', core_results(v).weight_entropy));
    end

end

Table6_Core_Ablation = table( ...
    Variant, ...
    IGD, ...
    Spread, ...
    TD_error_Var, ...
    Weight_Entropy, ...
    Stability);

%% ------------------------------------------------------------
%  10. Build Table 7: Dynamic weighting behavior
%% ------------------------------------------------------------

fprintf('\nBuilding Table 7 dynamic weighting behavior...\n');

idx_fixed = find(strcmp({core_results.variant_name}, 'Fixed weights'));
idx_utility = find(strcmp({core_results.variant_name}, 'Utility-proportional weights'));
idx_deficiency = find(strcmp({core_results.variant_name}, 'RGN-MARL full'));

idx_table7 = [idx_fixed, idx_utility, idx_deficiency];

Weighting_rule = [
    "Fixed weights"
    "Utility-proportional weights"
    "Deficiency-driven weights"
];

Avg_entropy = zeros(3, 1);
Entropy_std = zeros(3, 1);
Collapse_ratio = zeros(3, 1);

for ii = 1:3

    v = idx_table7(ii);

    Avg_entropy(ii) = core_results(v).weight_entropy;
    Entropy_std(ii) = core_results(v).entropy_std;
    Collapse_ratio(ii) = core_results(v).collapse_ratio;

end

Table7_Dynamic_Weighting = table( ...
    Weighting_rule, ...
    Avg_entropy, ...
    Entropy_std, ...
    Collapse_ratio);

%% ------------------------------------------------------------
%  11. Run Table 8: K_n = 1, 2, 3, 4
%% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('Running Table 8 negotiation-round ablation...\n');
fprintf('============================================================\n');

round_list = [1, 2, 3, 4];
num_rounds = length(round_list);

round_results_cell = cell(num_rounds, 1);

for k = 1:num_rounds

    rng(2026);

    cfg = struct();

    cfg.name = sprintf('K_n=%d', round_list(k));
    cfg.data_mode = 'ccsd';
    cfg.use_confidence_replay = true;
    cfg.weight_mode = 'deficiency';
    cfg.max_rounds = round_list(k);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('Running negotiation-round variant: K_n = %d\n', round_list(k));
    fprintf('------------------------------------------------------------\n');

    round_results_cell{k} = run_single_variant(cfg, data_ccsd, params);


end
round_results = concatenate_structs_safely(round_results_cell);

%% ------------------------------------------------------------
%  12. Compute IGD and Spread for Table 8
%% ------------------------------------------------------------

fprintf('\nComputing IGD and Spread for Table 8...\n');

all_eval_Y_round = [];

for k = 1:num_rounds
    all_eval_Y_round = [all_eval_Y_round; round_results(k).eval_Y];
end

C_all_round = objective_to_cost(all_eval_Y_round);
nd_idx_round = get_nondominated_indices(C_all_round);
Y_ref_round = all_eval_Y_round(nd_idx_round, :);

K_n = round_list(:);
IGD_round = zeros(num_rounds, 1);
Spread_round = zeros(num_rounds, 1);
Disagreement = zeros(num_rounds, 1);
Latency_ms = zeros(num_rounds, 1);

for k = 1:num_rounds

    round_results(k).IGD = compute_igd(Y_ref_round, round_results(k).eval_Y);
    round_results(k).Spread = compute_spread(round_results(k).eval_Y);

    IGD_round(k) = round_results(k).IGD;
    Spread_round(k) = round_results(k).Spread;
    Disagreement(k) = round_results(k).mean_disagreement_after;
    Latency_ms(k) = round_results(k).mean_latency_ms;

end

Table8_NGAF_Rounds = table( ...
    K_n, ...
    IGD_round, ...
    Spread_round, ...
    Disagreement, ...
    Latency_ms);

%% ------------------------------------------------------------
%  13. Display tables
%% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('Table 6. Core mechanism-attribution ablation study\n');
fprintf('============================================================\n');
disp(Table6_Core_Ablation);

fprintf('\n============================================================\n');
fprintf('Table 7. Dynamic weighting behavior\n');
fprintf('============================================================\n');
disp(Table7_Dynamic_Weighting);

fprintf('\n============================================================\n');
fprintf('Table 8. Effect of negotiation rounds on coordination and latency\n');
fprintf('============================================================\n');
disp(Table8_NGAF_Rounds);

%% ------------------------------------------------------------
%  14. Save results
%% ------------------------------------------------------------

save(result_file, ...
    'Table6_Core_Ablation', ...
    'Table7_Dynamic_Weighting', ...
    'Table8_NGAF_Rounds', ...
    'core_results', ...
    'round_results', ...
    'core_variants', ...
    'round_list', ...
    'Y_ref_core', ...
    'Y_ref_round', ...
    'data_ccsd', ...
    'data_raw', ...
    'params', ...
    'idx_ccsd', ...
    'idx_raw', ...
    'target_segment_id');

fprintf('\nAll ablation table results saved to %s.\n', result_file);

%% ------------------------------------------------------------
%  15. Simple visualization
%% ------------------------------------------------------------

figure('Name', 'Table 6 Core Ablation Metrics', ...
    'Position', [100, 100, 1200, 700]);

subplot(2, 2, 1);
bar(IGD);
set(gca, 'XTick', 1:num_core_variants, 'XTickLabel', Variant);
xtickangle(25);
ylabel('IGD');
title('IGD ↓');
grid on;

subplot(2, 2, 2);
bar(Spread);
set(gca, 'XTick', 1:num_core_variants, 'XTickLabel', Variant);
xtickangle(25);
ylabel('Spread');
title('Spread ↓');
grid on;

subplot(2, 2, 3);
bar(TD_error_Var);
set(gca, 'XTick', 1:num_core_variants, 'XTickLabel', Variant);
xtickangle(25);
ylabel('TD-error variance');
title('TD-error Var. ↓');
grid on;

subplot(2, 2, 4);
bar(Stability);
set(gca, 'XTick', 1:num_core_variants, 'XTickLabel', Variant);
xtickangle(25);
ylabel('Stability');
title('Stability ↑');
grid on;

sgtitle('Table 6 Core Ablation Metrics');

figure('Name', 'Table 8 Negotiation Rounds', ...
    'Position', [150, 150, 1000, 500]);

subplot(1, 2, 1);
plot(K_n, IGD_round, '-o', 'LineWidth', 1.8);
xlabel('K_n');
ylabel('IGD');
title('IGD vs. Negotiation Rounds');
grid on;

subplot(1, 2, 2);
plot(K_n, Latency_ms, '-o', 'LineWidth', 1.8);
xlabel('K_n');
ylabel('Latency / ms');
title('Latency vs. Negotiation Rounds');
grid on;

%% ============================================================
%  Local functions
%% ============================================================

function data_source = prepare_data_source(X, Y, idx, params, source_name)

    X_seg = X(idx, :);
    Y_seg = Y(idx, :);

    n_seg = size(X_seg, 1);
    n_train = floor(params.train_ratio * n_seg);

    train_idx = 1:n_train;
    test_idx  = n_train+1:n_seg;

    X_train = X_seg(train_idx, :);
    Y_train = Y_seg(train_idx, :);

    X_test = X_seg(test_idx, :);
    Y_test = Y_seg(test_idx, :);

    fprintf('Training surrogate models for %s...\n', source_name);

    model_y1 = fitrgp(X_train, Y_train(:, 1), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true);

    model_y2 = fitrgp(X_train, Y_train(:, 2), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true);

    model_y3 = fitrgp(X_train, Y_train(:, 3), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true);

    lb = min(X_train, [], 1);
    ub = max(X_train, [], 1);

    for j = 1:size(X_train, 2)
        if abs(ub(j) - lb(j)) < 1e-8
            lb(j) = lb(j) - 1e-4;
            ub(j) = ub(j) + 1e-4;
        end
    end

    action_range = ub - lb;

    fprintf('Generating candidate pool for %s...\n', source_name);

    U_pool = lhsdesign(params.num_candidates, size(X_train, 2));
    U_pool = lb + U_pool .* action_range;

    Pool_Y = predict_three_objectives(U_pool, model_y1, model_y2, model_y3);
    Pool_Utility = convert_to_utility_matrix(Pool_Y);

    confidence = compute_segment_confidence(X_train, Y_train);

    candidate_confidence = map_confidence_to_candidates( ...
        U_pool, X_train, confidence);

    train_dist = pairwise_knn_distance(X_train, X_train, params.knn_k + 1);
    train_dist = train_dist(:, end);

    D_th = quantile(train_dist, params.ood_threshold_quantile);

    data_source = struct();

    data_source.source_name = source_name;

    data_source.X_train = X_train;
    data_source.Y_train = Y_train;
    data_source.X_test = X_test;
    data_source.Y_test = Y_test;

    data_source.model_y1 = model_y1;
    data_source.model_y2 = model_y2;
    data_source.model_y3 = model_y3;

    data_source.lb = lb;
    data_source.ub = ub;
    data_source.action_range = action_range;

    data_source.U_pool = U_pool;
    data_source.Pool_Y = Pool_Y;
    data_source.Pool_Utility = Pool_Utility;

    data_source.confidence = confidence;
    data_source.candidate_confidence = candidate_confidence;

    data_source.D_th = D_th;

end

function result = run_single_variant(cfg, data, params)

    num_agents = params.num_agents;
    num_features = params.num_features;
    num_candidates = params.num_candidates;

    X_train = data.X_train;
    Y_train = data.Y_train;
    X_test = data.X_test;
    Y_test = data.Y_test;

    model_y1 = data.model_y1;
    model_y2 = data.model_y2;
    model_y3 = data.model_y3;

    lb = data.lb;
    ub = data.ub;
    action_range = data.action_range;

    U_pool = data.U_pool;
    Pool_Y = data.Pool_Y;
    Pool_Utility = data.Pool_Utility;

    candidate_confidence = data.candidate_confidence;
    D_th = data.D_th;

    if cfg.use_confidence_replay
        replay_prob = candidate_confidence + 1e-6;
        replay_prob = replay_prob ./ sum(replay_prob);
    else
        replay_prob = ones(num_candidates, 1) ./ num_candidates;
    end

    actor_actions = zeros(num_agents, num_features);

    for i = 1:num_agents
        [~, best_i_idx] = max(Pool_Utility(:, i));
        actor_actions(i, :) = U_pool(best_i_idx, :);
    end

    smoothed_utility = mean(Pool_Utility, 1);
    authority_weights = ones(1, num_agents) / num_agents;

    authority_history = zeros(params.num_epochs, num_agents);
    deficiency_history = zeros(params.num_epochs, num_agents);
    utility_history = zeros(params.num_epochs, num_agents);

    global_reward_history = zeros(params.num_epochs, 1);
    ood_history = zeros(params.num_epochs, 1);
    ngaf_round_history = zeros(params.num_epochs, 1);
    latency_history_ms = zeros(params.num_epochs, 1);

    disagreement_before_history = zeros(params.num_epochs, 1);
    disagreement_after_history  = zeros(params.num_epochs, 1);

    sampled_confidence_history = zeros(params.num_epochs, 1);

    exploration_noise = params.exploration_noise_init;

    for epoch = 1:params.num_epochs

        batch_idx = randsample(num_candidates, params.batch_size, true, replay_prob);

        Utility_batch = Pool_Utility(batch_idx, :);

        sampled_confidence_history(epoch) = mean(candidate_confidence(batch_idx));

        proposals = zeros(num_agents, num_features);

        for i = 1:num_agents
            noise = exploration_noise * randn(1, num_features) .* action_range;
            proposals(i, :) = actor_actions(i, :) + noise;
            proposals(i, :) = min(max(proposals(i, :), lb), ub);
        end

        current_utilities = zeros(1, num_agents);

        for i = 1:num_agents
            prop_y = predict_three_objectives(proposals(i, :), model_y1, model_y2, model_y3);
            prop_u = convert_to_utility_matrix_with_ref(prop_y, Pool_Y);
            current_utilities(i) = prop_u(i);
        end

        smoothed_utility = ...
            params.utility_smoothing_lambda * current_utilities + ...
            (1 - params.utility_smoothing_lambda) * smoothed_utility;

        deficiency = 1 - smoothed_utility;

        authority_weights = compute_authority_weights( ...
            cfg.weight_mode, ...
            deficiency, ...
            smoothed_utility, ...
            authority_weights, ...
            params.authority_temperature, ...
            params.authority_min_weight, ...
            params.authority_smoothing_alpha);

        shared_before = sum(proposals .* authority_weights', 1);

        disagreement_before_history(epoch) = compute_proposal_disagreement( ...
            proposals, shared_before);

        tic;
        [shared_action, final_proposals, actual_rounds] = ngaf_fusion( ...
            proposals, ...
            authority_weights, ...
            model_y1, model_y2, model_y3, ...
            Pool_Y, ...
            X_train, ...
            lb, ub, ...
            cfg.max_rounds, ...
            params.ngaf_step_size, ...
            params.consensus_lambda, ...
            params.ood_gamma, ...
            D_th, ...
            params.knn_k, ...
            params.ngaf_converge_tol);
        latency_history_ms(epoch) = toc * 1000;

        disagreement_after_history(epoch) = compute_proposal_disagreement( ...
            final_proposals, shared_action);

        shared_y = predict_three_objectives(shared_action, model_y1, model_y2, model_y3);
        shared_utility = convert_to_utility_matrix_with_ref(shared_y, Pool_Y);

        ood_penalty = compute_ood_penalty(shared_action, X_train, D_th, params.knn_k);

        agent_rewards = zeros(1, num_agents);

        for i = 1:num_agents

            consensus_penalty = norm(final_proposals(i, :) - shared_action, 2)^2;

            agent_rewards(i) = ...
                shared_utility(i) ...
                - params.beta_consensus * consensus_penalty ...
                - params.gamma_ood_reward * ood_penalty;

        end

        global_reward = sum(authority_weights .* agent_rewards);

        for i = 1:num_agents

            [~, rank_idx] = sort(Utility_batch(:, i), 'descend');

            elite_num = max(3, round(0.25 * params.batch_size));
            elite_idx = batch_idx(rank_idx(1:elite_num));

            target_action_i = mean(U_pool(elite_idx, :), 1);

            proposals_minus_i = final_proposals;
            proposals_minus_i(i, :) = mean(final_proposals(setdiff(1:num_agents, i), :), 1);

            u_minus_i = sum(proposals_minus_i .* authority_weights', 1);
            y_minus_i = predict_three_objectives(u_minus_i, model_y1, model_y2, model_y3);
            utility_minus_i = convert_to_utility_matrix_with_ref(y_minus_i, Pool_Y);

            reward_minus_i = sum(authority_weights .* utility_minus_i);
            cf_advantage = global_reward - reward_minus_i;

            step_size = params.actor_learning_rate * authority_weights(i) * max(0.1, 1 + cf_advantage);

            actor_actions(i, :) = ...
                (1 - step_size) * actor_actions(i, :) + ...
                step_size * target_action_i;

            actor_actions(i, :) = min(max(actor_actions(i, :), lb), ub);

        end

        exploration_noise = exploration_noise * params.exploration_noise_decay;

        authority_history(epoch, :) = authority_weights;
        deficiency_history(epoch, :) = deficiency;
        utility_history(epoch, :) = smoothed_utility;

        global_reward_history(epoch) = global_reward;
        ood_history(epoch) = ood_penalty;
        ngaf_round_history(epoch) = actual_rounds;

        if mod(epoch, 30) == 0 || epoch == 1
            fprintf(['%s | Epoch %3d | Reward %.4f | OOD %.4f | ', ...
                     'Dis %.4f -> %.4f | Lat %.3f ms\n'], ...
                cfg.name, ...
                epoch, ...
                global_reward, ...
                ood_penalty, ...
                disagreement_before_history(epoch), ...
                disagreement_after_history(epoch), ...
                latency_history_ms(epoch));
        end

    end

    eval_U = zeros(params.num_eval_rollouts, num_features);
    eval_Y = zeros(params.num_eval_rollouts, params.num_objectives);
    eval_authority = zeros(params.num_eval_rollouts, num_agents);
    eval_ood = zeros(params.num_eval_rollouts, 1);
    eval_ngaf_rounds = zeros(params.num_eval_rollouts, 1);
    eval_latency_ms = zeros(params.num_eval_rollouts, 1);

    eval_disagreement_before = zeros(params.num_eval_rollouts, 1);
    eval_disagreement_after  = zeros(params.num_eval_rollouts, 1);

    for r = 1:params.num_eval_rollouts

        proposals = zeros(num_agents, num_features);

        for i = 1:num_agents
            noise = 0.02 * randn(1, num_features) .* action_range;
            proposals(i, :) = actor_actions(i, :) + noise;
            proposals(i, :) = min(max(proposals(i, :), lb), ub);
        end

        eval_utilities = zeros(1, num_agents);

        for i = 1:num_agents
            prop_y = predict_three_objectives(proposals(i, :), model_y1, model_y2, model_y3);
            prop_u = convert_to_utility_matrix_with_ref(prop_y, Pool_Y);
            eval_utilities(i) = prop_u(i);
        end

        eval_deficiency = 1 - eval_utilities;

        eval_w = compute_authority_weights( ...
            cfg.weight_mode, ...
            eval_deficiency, ...
            eval_utilities, ...
            authority_weights, ...
            params.authority_temperature, ...
            params.authority_min_weight, ...
            1.0);

        u_before_eval = sum(proposals .* eval_w', 1);

        eval_disagreement_before(r) = compute_proposal_disagreement( ...
            proposals, u_before_eval);

        tic;
        [u_eval, final_eval_proposals, rounds_eval] = ngaf_fusion( ...
            proposals, ...
            eval_w, ...
            model_y1, model_y2, model_y3, ...
            Pool_Y, ...
            X_train, ...
            lb, ub, ...
            cfg.max_rounds, ...
            params.ngaf_step_size, ...
            params.consensus_lambda, ...
            params.ood_gamma, ...
            D_th, ...
            params.knn_k, ...
            params.ngaf_converge_tol);
        eval_latency_ms(r) = toc * 1000;

        y_eval = predict_three_objectives(u_eval, model_y1, model_y2, model_y3);

        eval_U(r, :) = u_eval;
        eval_Y(r, :) = y_eval;
        eval_authority(r, :) = eval_w;
        eval_ood(r) = compute_ood_penalty(u_eval, X_train, D_th, params.knn_k);
        eval_ngaf_rounds(r) = rounds_eval;

        eval_disagreement_after(r) = compute_proposal_disagreement( ...
            final_eval_proposals, u_eval);

    end

    best_idx = select_by_ideal_distance(eval_Y);

    test_pred_y = predict_three_objectives(X_test, model_y1, model_y2, model_y3);

    surrogate_mae = mean(abs(test_pred_y - Y_test), 1);
    surrogate_rmse = sqrt(mean((test_pred_y - Y_test).^2, 1));

    result = struct();

    result.variant_name = cfg.name;
    result.data_source = data.source_name;
    result.use_confidence_replay = cfg.use_confidence_replay;
    result.weight_mode = cfg.weight_mode;
    result.max_rounds = cfg.max_rounds;

    result.actor_actions = actor_actions;

    result.eval_U = eval_U;
    result.eval_Y = eval_Y;
    result.eval_authority = eval_authority;
    result.eval_ood = eval_ood;
    result.eval_ngaf_rounds = eval_ngaf_rounds;
    result.eval_latency_ms = eval_latency_ms;

    result.eval_disagreement_before = eval_disagreement_before;
    result.eval_disagreement_after = eval_disagreement_after;

    result.best_idx = best_idx;
    result.best_u = eval_U(best_idx, :);
    result.best_y = eval_Y(best_idx, :);

    result.authority_history = authority_history;
    result.deficiency_history = deficiency_history;
    result.utility_history = utility_history;

    result.global_reward_history = global_reward_history;
    result.ood_history = ood_history;
    result.ngaf_round_history = ngaf_round_history;
    result.latency_history_ms = latency_history_ms;

    result.disagreement_before_history = disagreement_before_history;
    result.disagreement_after_history = disagreement_after_history;

    result.sampled_confidence_history = sampled_confidence_history;

    result.mean_disagreement_before = mean(eval_disagreement_before);
    result.mean_disagreement_after = mean(eval_disagreement_after);
    result.mean_latency_ms = mean(eval_latency_ms);
    result.mean_ood = mean(eval_ood);
    result.mean_ngaf_rounds = mean(eval_ngaf_rounds);

    result.surrogate_mae = surrogate_mae;
    result.surrogate_rmse = surrogate_rmse;

end

function authority_weights = compute_authority_weights( ...
    weight_mode, ...
    deficiency, ...
    utility, ...
    previous_weights, ...
    temperature, ...
    min_weight, ...
    smoothing_alpha)

    num_agents = length(deficiency);

    switch weight_mode

        case 'deficiency'

            raw_weights = softmax_vector(deficiency(:) / temperature)';

            corrected_weights = max(raw_weights, min_weight);
            corrected_weights = corrected_weights ./ sum(corrected_weights);

            authority_weights = ...
                (1 - smoothing_alpha) * previous_weights + ...
                smoothing_alpha * corrected_weights;

            authority_weights = authority_weights ./ sum(authority_weights);

        case 'fixed'

            authority_weights = ones(1, num_agents) / num_agents;

        case 'utility'

            raw_weights = utility(:)';
            raw_weights = max(raw_weights, 1e-6);
            raw_weights = raw_weights ./ sum(raw_weights);

            corrected_weights = max(raw_weights, min_weight);
            corrected_weights = corrected_weights ./ sum(corrected_weights);

            authority_weights = ...
                (1 - smoothing_alpha) * previous_weights + ...
                smoothing_alpha * corrected_weights;

            authority_weights = authority_weights ./ sum(authority_weights);

        otherwise
            error('Unknown weight mode: %s', weight_mode);

    end

end

function Y = predict_three_objectives(U, model_y1, model_y2, model_y3)

    y1 = predict(model_y1, U);
    y2 = predict(model_y2, U);
    y3 = predict(model_y3, U);

    Y = [y1, y2, y3];

end

function Utility = convert_to_utility_matrix(Y)

    y1 = Y(:, 1);
    y2 = Y(:, 2);
    y3 = Y(:, 3);

    u1 = (max(y1) - y1) ./ (max(y1) - min(y1) + eps);
    u2 = (max(y2) - y2) ./ (max(y2) - min(y2) + eps);
    u3 = (y3 - min(y3)) ./ (max(y3) - min(y3) + eps);

    Utility = [u1, u2, u3];
    Utility = min(max(Utility, 0), 1);

end

function Utility = convert_to_utility_matrix_with_ref(Y, Y_ref)

    y1 = Y(:, 1);
    y2 = Y(:, 2);
    y3 = Y(:, 3);

    ref1 = Y_ref(:, 1);
    ref2 = Y_ref(:, 2);
    ref3 = Y_ref(:, 3);

    u1 = (max(ref1) - y1) ./ (max(ref1) - min(ref1) + eps);
    u2 = (max(ref2) - y2) ./ (max(ref2) - min(ref2) + eps);
    u3 = (y3 - min(ref3)) ./ (max(ref3) - min(ref3) + eps);

    Utility = [u1, u2, u3];
    Utility = min(max(Utility, 0), 1);

end

function confidence = compute_segment_confidence(X_train, Y_train)

    Z = [normalize_matrix(X_train), normalize_matrix(Y_train)];

    center = median(Z, 1);
    dist = sqrt(sum((Z - center).^2, 2));

    sigma2 = var(dist) + eps;

    confidence = exp(-(dist.^2) ./ sigma2);
    confidence = confidence(:);

    confidence = confidence ./ (max(confidence) + eps);
    confidence = max(confidence, 1e-4);

end

function Z = normalize_matrix(X)

    X_min = min(X, [], 1);
    X_max = max(X, [], 1);

    Z = (X - X_min) ./ (X_max - X_min + eps);

end

function cand_conf = map_confidence_to_candidates(U_pool, X_train, confidence)

    num_candidates = size(U_pool, 1);
    cand_conf = zeros(num_candidates, 1);

    for i = 1:num_candidates

        diff = X_train - U_pool(i, :);
        dist = sqrt(sum(diff.^2, 2));

        [~, idx_min] = min(dist);
        cand_conf(i) = confidence(idx_min);

    end

    cand_conf = cand_conf ./ (max(cand_conf) + eps);
    cand_conf = max(cand_conf, 1e-4);

end

function d = pairwise_knn_distance(A, B, k)

    nA = size(A, 1);
    d = zeros(nA, k);

    for i = 1:nA

        diff = B - A(i, :);
        dist = sqrt(sum(diff.^2, 2));
        dist = sort(dist, 'ascend');

        k_eff = min(k, length(dist));

        d(i, 1:k_eff) = dist(1:k_eff);

        if k_eff < k
            d(i, k_eff+1:k) = dist(end);
        end

    end

end

function penalty = compute_ood_penalty(u, X_train, D_th, knn_k)

    diff = X_train - u;
    dist = sqrt(sum(diff.^2, 2));
    dist = sort(dist, 'ascend');

    k_idx = min(knn_k, length(dist));
    Dk = dist(k_idx);

    penalty = max(0, Dk - D_th);

end

function disagreement = compute_proposal_disagreement(proposals, shared_action)

    num_agents = size(proposals, 1);
    dist_sum = 0;

    for i = 1:num_agents
        dist_sum = dist_sum + norm(proposals(i, :) - shared_action, 2);
    end

    disagreement = dist_sum / num_agents;

end

function [shared_action, proposals, actual_rounds] = ngaf_fusion( ...
    proposals, ...
    authority_weights, ...
    model_y1, model_y2, model_y3, ...
    Pool_Y, ...
    X_train, ...
    lb, ub, ...
    max_rounds, ...
    base_step_size, ...
    consensus_lambda, ...
    ood_gamma, ...
    D_th, ...
    knn_k, ...
    converge_tol)

    num_agents = size(proposals, 1);

    prev_shared = sum(proposals .* authority_weights', 1);
    shared_action = prev_shared;

    actual_rounds = max_rounds;

    if max_rounds <= 1

        shared_action = sum(proposals .* authority_weights', 1);
        shared_action = min(max(shared_action, lb), ub);

        return;

    end

    for k = 1:max_rounds

        shared_action = sum(proposals .* authority_weights', 1);

        ood_penalty = compute_ood_penalty(shared_action, X_train, D_th, knn_k);
        effective_step = base_step_size * exp(-ood_gamma * ood_penalty);

        new_proposals = proposals;

        for i = 1:num_agents

            g_i = estimate_improvement_direction( ...
                proposals(i, :), ...
                i, ...
                model_y1, model_y2, model_y3, ...
                Pool_Y, ...
                lb, ub);

            consensus_deviation = proposals(i, :) - shared_action;

            new_proposals(i, :) = ...
                proposals(i, :) + ...
                effective_step * g_i - ...
                consensus_lambda * consensus_deviation;

            new_proposals(i, :) = min(max(new_proposals(i, :), lb), ub);

        end

        proposals = new_proposals;

        new_shared = sum(proposals .* authority_weights', 1);

        if norm(new_shared - prev_shared, 2) < converge_tol

            actual_rounds = k;
            shared_action = new_shared;

            break;

        end

        prev_shared = new_shared;

    end

    shared_action = min(max(shared_action, lb), ub);

end

function g = estimate_improvement_direction( ...
    u, ...
    agent_id, ...
    model_y1, model_y2, model_y3, ...
    Pool_Y, ...
    lb, ub)

    num_features = length(u);
    g = zeros(1, num_features);

    step = 1e-3 * (ub - lb + eps);

    for j = 1:num_features

        u_plus = u;
        u_minus = u;

        u_plus(j) = min(ub(j), u_plus(j) + step(j));
        u_minus(j) = max(lb(j), u_minus(j) - step(j));

        y_plus = predict_three_objectives(u_plus, model_y1, model_y2, model_y3);
        y_minus = predict_three_objectives(u_minus, model_y1, model_y2, model_y3);

        utility_plus = convert_to_utility_matrix_with_ref(y_plus, Pool_Y);
        utility_minus = convert_to_utility_matrix_with_ref(y_minus, Pool_Y);

        g(j) = (utility_plus(agent_id) - utility_minus(agent_id)) ...
               / (u_plus(j) - u_minus(j) + eps);

    end

    g_norm = norm(g, 2);

    if g_norm > 0
        g = g ./ g_norm;
    end

end

function w = softmax_vector(x)

    x = x(:);
    x = x - max(x);

    exp_x = exp(x);

    w = exp_x ./ (sum(exp_x) + eps);

end

function best_idx = select_by_ideal_distance(Y)

    y1 = Y(:, 1);
    y2 = Y(:, 2);
    y3 = Y(:, 3);

    c1 = (y1 - min(y1)) ./ (max(y1) - min(y1) + eps);
    c2 = (y2 - min(y2)) ./ (max(y2) - min(y2) + eps);
    c3 = (max(y3) - y3) ./ (max(y3) - min(y3) + eps);

    C = [c1, c2, c3];

    dist = sqrt(sum(C.^2, 2));

    [~, best_idx] = min(dist);

end

function C = objective_to_cost(Y)

    % Convert mixed-direction objectives into minimization costs.
    % Y1: minimize
    % Y2: minimize
    % Y3: maximize, therefore use -Y3.

    C = [Y(:, 1), Y(:, 2), -Y(:, 3)];

end

function Cn = normalize_cost_with_ref(C, C_ref)

    c_min = min(C_ref, [], 1);
    c_max = max(C_ref, [], 1);

    Cn = (C - c_min) ./ (c_max - c_min + eps);

end

function nd_idx = get_nondominated_indices(C)

    % C is a minimization cost matrix.

    n = size(C, 1);
    is_dominated = false(n, 1);

    for i = 1:n

        for j = 1:n

            if i ~= j

                if all(C(j, :) <= C(i, :)) && any(C(j, :) < C(i, :))
                    is_dominated(i) = true;
                    break;
                end

            end

        end

    end

    nd_idx = find(~is_dominated);

end

function igd_value = compute_igd(Y_ref, Y_method)

    C_ref = objective_to_cost(Y_ref);
    C_method = objective_to_cost(Y_method);

    C_all = [C_ref; C_method];

    C_ref_n = normalize_cost_with_ref(C_ref, C_all);
    C_method_n = normalize_cost_with_ref(C_method, C_all);

    n_ref = size(C_ref_n, 1);
    dist_min = zeros(n_ref, 1);

    for i = 1:n_ref

        diff = C_method_n - C_ref_n(i, :);
        dist = sqrt(sum(diff.^2, 2));

        dist_min(i) = min(dist);

    end

    igd_value = mean(dist_min);

end

function spread_value = compute_spread(Y_method)

    C = objective_to_cost(Y_method);
    Cn = normalize_cost_with_ref(C, C);

    n = size(Cn, 1);

    if n <= 2
        spread_value = 0;
        return;
    end

    nn_dist = zeros(n, 1);

    for i = 1:n

        diff = Cn - Cn(i, :);
        dist = sqrt(sum(diff.^2, 2));
        dist(i) = inf;

        nn_dist(i) = min(dist);

    end

    spread_value = std(nn_dist);

end

function [avg_entropy, entropy_std, collapse_ratio] = compute_weight_entropy_metrics( ...
    authority_history, collapse_threshold)

    W = authority_history;

    W = max(W, eps);
    W = W ./ sum(W, 2);

    num_agents = size(W, 2);

    H = -sum(W .* log(W), 2) ./ log(num_agents);

    avg_entropy = mean(H);
    entropy_std = std(H);

    collapse_ratio = mean(max(W, [], 2) > collapse_threshold);

end

function td_var = compute_td_error_variance(global_reward_history)

    r = global_reward_history(:);

    if length(r) <= 2
        td_var = 0;
        return;
    end

    td_error = r(2:end) - r(1:end-1);

    td_var = var(td_error);

end

function stability = compute_training_stability(global_reward_history)

    r = global_reward_history(:);

    tail_len = min(20, length(r));
    r_tail = r(end-tail_len+1:end);

    stability = exp(-std(r_tail));

end
function S = concatenate_structs_safely(S_cell)
    % Concatenate a cell array of structs into a struct array.
    % If some structs have missing fields, fill them with [].

    n = numel(S_cell);

    empty_flags = cellfun(@isempty, S_cell);

    if any(empty_flags)
        disp('Empty struct cells detected at indices:');
        disp(find(empty_flags));
        error('Cannot concatenate because some result cells are empty.');
    end

    all_fields = {};

    for i = 1:n
        f = fieldnames(S_cell{i});
        all_fields = union(all_fields, f);
    end

    empty_template = struct();

    for j = 1:numel(all_fields)
        empty_template.(all_fields{j}) = [];
    end

    S = repmat(empty_template, n, 1);

    for i = 1:n
        f = fieldnames(S_cell{i});

        for j = 1:numel(f)
            S(i).(f{j}) = S_cell{i}.(f{j});
        end
    end
end

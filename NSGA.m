%% ============================================================
%  NSGA-II Baseline for Table 5
%
%  Output format:
%    Method | Regime | MEAN(Y1) | MEAN(Y2) | MEAN(Y3)
%           | IGD | HV | Coverage | Spread | Stability
%
%  Each regime is reported separately:
%    Segment 1 -> R_H
%    Segment 2 -> R_M
%    Segment 3 -> R_L
%
%  Objectives:
%    Y1: unit heat consumption, minimize
%    Y2: unit electricity consumption, minimize
%    Y3: clinker output, maximize
%
%  Note:
%    This implementation uses MATLAB gamultiobj as NSGA-II baseline.
%% ============================================================

clear; clc; close all;

%% ------------------------------------------------------------
%  1. Basic settings
%% ------------------------------------------------------------

rng(2026);

data_path = './';

file_x  = fullfile(data_path, 'xselect.xlsx');
file_y1 = fullfile(data_path, 'y1select.xlsx');
file_y2 = fullfile(data_path, 'y2select.xlsx');
file_y3 = fullfile(data_path, 'y3select.xlsx');

segment_file = 'selected_segments.mat';
result_file  = 'results_nsga2.mat';

regime_labels = ["R_H"; "R_M"; "R_L"];

train_ratio = 0.7;

% Repeated runs for robustness / stability
num_runs = 10;

% NSGA-II settings
population_size = 80;
max_generations = 100;

% Reference front construction
num_ref_candidates = 20000;

% Hypervolume settings
hv_ref_point = [1.1, 1.1, 1.1];
hv_mc_samples = 50000;

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

fprintf('Valid samples: %d\n', num_samples);
fprintf('Input variables: %d\n', num_features);

if length(Y1) ~= num_samples || length(Y2) ~= num_samples || length(Y3) ~= num_samples
    error('Data length mismatch among X, Y1, Y2, and Y3.');
end

%% ------------------------------------------------------------
%  3. Load common selected regimes
%% ------------------------------------------------------------

if ~exist(segment_file, 'file')
    error('selected_segments.mat not found. Please run Weighted Sum baseline first.');
end

load(segment_file, 'segments', 'segment_len', 'num_segments');

if num_segments ~= 3
    error('Table 5 requires exactly three regimes: R_H, R_M, and R_L.');
end

disp('Loaded regime index ranges:');

for s = 1:num_segments
    fprintf('%s: %d -- %d\n', regime_labels(s), segments{s}(1), segments{s}(end));
end

%% ------------------------------------------------------------
%  4. Run NSGA-II baseline for each regime
%% ------------------------------------------------------------

results = struct();

for s = 1:num_segments

    fprintf('\n============================================================\n');
    fprintf('NSGA-II on Regime %s\n', regime_labels(s));
    fprintf('============================================================\n');

    idx = segments{s};

    X_seg = X(idx, :);
    Y_seg = Y(idx, :);

    n_seg = size(X_seg, 1);
    n_train = floor(train_ratio * n_seg);

    train_idx_local = 1:n_train;
    test_idx_local  = n_train+1:n_seg;

    X_train = X_seg(train_idx_local, :);
    Y_train = Y_seg(train_idx_local, :);

    X_test = X_seg(test_idx_local, :);
    Y_test = Y_seg(test_idx_local, :);

    %% --------------------------------------------------------
    %  4.1 Train local surrogate models
    %% --------------------------------------------------------

    fprintf('Training local surrogate models...\n');

    model_y1 = fitrgp(X_train, Y_train(:, 1), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true);

    model_y2 = fitrgp(X_train, Y_train(:, 2), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true);

    model_y3 = fitrgp(X_train, Y_train(:, 3), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true);

    %% --------------------------------------------------------
    %  4.2 Surrogate test error
    %% --------------------------------------------------------

    test_pred_y1 = predict(model_y1, X_test);
    test_pred_y2 = predict(model_y2, X_test);
    test_pred_y3 = predict(model_y3, X_test);

    test_pred_Y = [test_pred_y1, test_pred_y2, test_pred_y3];

    surrogate_mae = mean(abs(test_pred_Y - Y_test), 1);
    surrogate_rmse = sqrt(mean((test_pred_Y - Y_test).^2, 1));

    fprintf('Surrogate MAE : [%.4f, %.4f, %.4f]\n', ...
        surrogate_mae(1), surrogate_mae(2), surrogate_mae(3));

    fprintf('Surrogate RMSE: [%.4f, %.4f, %.4f]\n', ...
        surrogate_rmse(1), surrogate_rmse(2), surrogate_rmse(3));

    %% --------------------------------------------------------
    %  4.3 Local action bounds
    %% --------------------------------------------------------

    lb = min(X_train, [], 1);
    ub = max(X_train, [], 1);

    for j = 1:num_features
        if abs(ub(j) - lb(j)) < 1e-10
            lb(j) = lb(j) - 1e-4;
            ub(j) = ub(j) + 1e-4;
        end
    end

    %% --------------------------------------------------------
    %  4.4 Build reference front for this regime
    %% --------------------------------------------------------

    rng(2026 + 1000 + s);

    U_ref_pool = lhsdesign(num_ref_candidates, num_features);
    U_ref_pool = lb + U_ref_pool .* (ub - lb);

    ref_y1 = predict(model_y1, U_ref_pool);
    ref_y2 = predict(model_y2, U_ref_pool);
    ref_y3 = predict(model_y3, U_ref_pool);

    Y_ref_pool = [ref_y1, ref_y2, ref_y3];

    C_ref_pool = objective_to_cost(Y_ref_pool);
    nd_ref_idx = get_nondominated_indices(C_ref_pool);

    Y_ref = Y_ref_pool(nd_ref_idx, :);

    fprintf('Reference front size for %s: %d\n', regime_labels(s), size(Y_ref, 1));

    %% --------------------------------------------------------
    %  4.5 Repeated NSGA-II runs
    %% --------------------------------------------------------

    run_results = struct();

    run_metric_matrix = zeros(num_runs, 7);

    for r = 1:num_runs

        rng(2026 + 100 * s + r);

        fprintf('\n%s | NSGA-II Run %02d/%02d\n', regime_labels(s), r, num_runs);

        objective_fun = @(u) nsga2_surrogate_objective( ...
            u, model_y1, model_y2, model_y3);

        options = optimoptions('gamultiobj', ...
            'PopulationSize', population_size, ...
            'MaxGenerations', max_generations, ...
            'Display', 'off', ...
            'UseParallel', false);

        [U_sol, F_sol] = gamultiobj( ...
            objective_fun, ...
            num_features, ...
            [], [], [], [], ...
            lb, ub, ...
            options);

        % F_sol is cost:
        %   [Y1, Y2, -Y3]
        % Convert back to original objective form.
        Y_sol = [F_sol(:, 1), F_sol(:, 2), -F_sol(:, 3)];

        % Remove duplicates.
        [U_sol, unique_idx] = unique(U_sol, 'rows', 'stable');
        Y_sol = Y_sol(unique_idx, :);

        % Keep nondominated solution set.
        C_sol = objective_to_cost(Y_sol);
        nd_sol_idx = get_nondominated_indices(C_sol);

        U_nd = U_sol(nd_sol_idx, :);
        Y_nd = Y_sol(nd_sol_idx, :);

        %% ----------------------------------------------------
        %  Metrics according to Table 5 definitions
        %% ----------------------------------------------------

        MEAN_Y = mean(Y_nd, 1);

        IGD_run = compute_igd(Y_ref, Y_nd);

        HV_run = compute_hv_monte_carlo_normalized( ...
            Y_nd, Y_ref, hv_ref_point, hv_mc_samples);

        Coverage_run = compute_reference_front_coverage(Y_nd, Y_ref);

        Spread_run = compute_spread(Y_nd);

        run_metric_matrix(r, :) = [
            MEAN_Y(1), ...
            MEAN_Y(2), ...
            -MEAN_Y(3), ...
            IGD_run, ...
            -HV_run, ...
            -Coverage_run, ...
            Spread_run];

        run_results(r).run_id = r;
        run_results(r).U_nd = U_nd;
        run_results(r).Y_nd = Y_nd;

        run_results(r).MEAN_Y1 = MEAN_Y(1);
        run_results(r).MEAN_Y2 = MEAN_Y(2);
        run_results(r).MEAN_Y3 = MEAN_Y(3);

        run_results(r).IGD = IGD_run;
        run_results(r).HV = HV_run;
        run_results(r).Coverage = Coverage_run;
        run_results(r).Spread = Spread_run;

        fprintf(['%s | Run %02d | MEAN(Y)=[%.4f %.4f %.4f] | ', ...
                 'IGD=%.5f | HV=%.5f | Coverage=%.5f | Spread=%.5f\n'], ...
            regime_labels(s), r, ...
            MEAN_Y(1), MEAN_Y(2), MEAN_Y(3), ...
            IGD_run, HV_run, Coverage_run, Spread_run);

    end

    %% --------------------------------------------------------
    %  4.6 Aggregate repeated runs for current regime
    %% --------------------------------------------------------

    MEAN_Y1_runs = [run_results.MEAN_Y1]';
    MEAN_Y2_runs = [run_results.MEAN_Y2]';
    MEAN_Y3_runs = [run_results.MEAN_Y3]';

    IGD_runs = [run_results.IGD]';
    HV_runs = [run_results.HV]';
    Coverage_runs = [run_results.Coverage]';
    Spread_runs = [run_results.Spread]';

    Stability_regime = compute_repeated_run_stability(run_metric_matrix);

    results(s).method_name = 'NSGA-II';
    results(s).regime = regime_labels(s);
    results(s).segment_id = s;
    results(s).index_range = [idx(1), idx(end)];

    results(s).train_idx = idx(train_idx_local);
    results(s).test_idx = idx(test_idx_local);

    results(s).Y_ref = Y_ref;
    results(s).run_results = run_results;

    results(s).MEAN_Y1 = mean(MEAN_Y1_runs);
    results(s).MEAN_Y2 = mean(MEAN_Y2_runs);
    results(s).MEAN_Y3 = mean(MEAN_Y3_runs);

    results(s).IGD = mean(IGD_runs);
    results(s).HV = mean(HV_runs);
    results(s).Coverage = mean(Coverage_runs);
    results(s).Spread = mean(Spread_runs);
    results(s).Stability = Stability_regime;

    results(s).STD_MEAN_Y1 = std(MEAN_Y1_runs);
    results(s).STD_MEAN_Y2 = std(MEAN_Y2_runs);
    results(s).STD_MEAN_Y3 = std(MEAN_Y3_runs);

    results(s).STD_IGD = std(IGD_runs);
    results(s).STD_HV = std(HV_runs);
    results(s).STD_Coverage = std(Coverage_runs);
    results(s).STD_Spread = std(Spread_runs);

    results(s).surrogate_mae = surrogate_mae;
    results(s).surrogate_rmse = surrogate_rmse;

    fprintf('\n%s summary:\n', regime_labels(s));
    fprintf('  MEAN(Y1)  : %.6f\n', results(s).MEAN_Y1);
    fprintf('  MEAN(Y2)  : %.6f\n', results(s).MEAN_Y2);
    fprintf('  MEAN(Y3)  : %.6f\n', results(s).MEAN_Y3);
    fprintf('  IGD       : %.6f\n', results(s).IGD);
    fprintf('  HV        : %.6f\n', results(s).HV);
    fprintf('  Coverage  : %.6f\n', results(s).Coverage);
    fprintf('  Spread    : %.6f\n', results(s).Spread);
    fprintf('  Stability : %.6f\n', results(s).Stability);

end

%% ------------------------------------------------------------
%  5. Build Table 5 format for NSGA-II
%% ------------------------------------------------------------

Method = strings(num_segments, 1);
Regime = strings(num_segments, 1);

MEAN_Y1 = zeros(num_segments, 1);
MEAN_Y2 = zeros(num_segments, 1);
MEAN_Y3 = zeros(num_segments, 1);

IGD = zeros(num_segments, 1);
HV = zeros(num_segments, 1);
Coverage = zeros(num_segments, 1);
Spread = zeros(num_segments, 1);
Stability = zeros(num_segments, 1);

for s = 1:num_segments

    Method(s) = "NSGA-II";
    Regime(s) = results(s).regime;

    MEAN_Y1(s) = results(s).MEAN_Y1;
    MEAN_Y2(s) = results(s).MEAN_Y2;
    MEAN_Y3(s) = results(s).MEAN_Y3;

    IGD(s) = results(s).IGD;
    HV(s) = results(s).HV;
    Coverage(s) = results(s).Coverage;
    Spread(s) = results(s).Spread;
    Stability(s) = results(s).Stability;

end

Table5_NSGAII = table( ...
    Method, ...
    Regime, ...
    MEAN_Y1, ...
    MEAN_Y2, ...
    MEAN_Y3, ...
    IGD, ...
    HV, ...
    Coverage, ...
    Spread, ...
    Stability);

fprintf('\n============================================================\n');
fprintf('Table 5 format: NSGA-II\n');
fprintf('============================================================\n');
disp(Table5_NSGAII);

%% ------------------------------------------------------------
%  6. Save results
%% ------------------------------------------------------------

save(result_file, ...
    'results', ...
    'Table5_NSGAII', ...
    'num_runs', ...
    'population_size', ...
    'max_generations', ...
    'num_ref_candidates', ...
    'hv_ref_point', ...
    'hv_mc_samples', ...
    'segments', ...
    'segment_len', ...
    'num_segments', ...
    'regime_labels');

fprintf('\nResults saved to %s.\n', result_file);

%% ------------------------------------------------------------
%  7. Visualization
%% ------------------------------------------------------------

figure('Name', 'NSGA-II Table 5 Metrics by Regime', ...
    'Position', [100, 100, 1400, 750]);

subplot(2, 4, 1);
bar(MEAN_Y1);
set(gca, 'XTickLabel', Regime);
ylabel('MEAN(Y1)');
title('MEAN(Y1) ↓');
grid on;

subplot(2, 4, 2);
bar(MEAN_Y2);
set(gca, 'XTickLabel', Regime);
ylabel('MEAN(Y2)');
title('MEAN(Y2) ↓');
grid on;

subplot(2, 4, 3);
bar(MEAN_Y3);
set(gca, 'XTickLabel', Regime);
ylabel('MEAN(Y3)');
title('MEAN(Y3) ↑');
grid on;

subplot(2, 4, 4);
bar(IGD);
set(gca, 'XTickLabel', Regime);
ylabel('IGD');
title('IGD ↓');
grid on;

subplot(2, 4, 5);
bar(HV);
set(gca, 'XTickLabel', Regime);
ylabel('HV');
title('HV ↑');
grid on;

subplot(2, 4, 6);
bar(Coverage);
set(gca, 'XTickLabel', Regime);
ylabel('Coverage');
title('Coverage ↑');
ylim([0, 1]);
grid on;

subplot(2, 4, 7);
bar(Spread);
set(gca, 'XTickLabel', Regime);
ylabel('Spread');
title('Spread ↓');
grid on;

subplot(2, 4, 8);
bar(Stability);
set(gca, 'XTickLabel', Regime);
ylabel('Stability');
title('Stability ↑');
ylim([0, 1]);
grid on;

sgtitle('NSGA-II Baseline Results for Table 5');

%% ============================================================
%  Local functions
%% ============================================================

function f = nsga2_surrogate_objective(u, model_y1, model_y2, model_y3)

    % gamultiobj minimizes all objectives.
    % Y1 minimize, Y2 minimize, Y3 maximize -> minimize -Y3.

    y1 = predict(model_y1, u);
    y2 = predict(model_y2, u);
    y3 = predict(model_y3, u);

    f = [y1, y2, -y3];

end

function C = objective_to_cost(Y)

    % Y1 minimize, Y2 minimize, Y3 maximize.
    C = [Y(:, 1), Y(:, 2), -Y(:, 3)];

end

function nd_idx = get_nondominated_indices(C)

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

function Cn = normalize_cost_with_ref(C, C_ref)

    c_min = min(C_ref, [], 1);
    c_max = max(C_ref, [], 1);

    Cn = (C - c_min) ./ (c_max - c_min + eps);

end

function igd_value = compute_igd(Y_ref, Y_method)

    if isempty(Y_ref) || isempty(Y_method)
        igd_value = nan;
        return;
    end

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

function hv = compute_hv_monte_carlo_normalized(Y_method, Y_ref, ref_point, num_mc)

    if isempty(Y_method) || isempty(Y_ref)
        hv = nan;
        return;
    end

    C_method = objective_to_cost(Y_method);
    C_ref = objective_to_cost(Y_ref);

    C_all = [C_method; C_ref];

    C_method_n = normalize_cost_with_ref(C_method, C_all);

    nd_idx = get_nondominated_indices(C_method_n);
    P = C_method_n(nd_idx, :);

    P = min(max(P, 0), ref_point);

    dim = size(P, 2);

    samples = rand(num_mc, dim) .* ref_point;

    dominated = false(num_mc, 1);

    for i = 1:size(P, 1)

        dominated_by_i = all(samples >= P(i, :), 2);
        dominated = dominated | dominated_by_i;

    end

    box_volume = prod(ref_point);

    hv = mean(dominated) * box_volume;

end

function coverage = compute_reference_front_coverage(Y_method, Y_ref)

    if isempty(Y_method) || isempty(Y_ref)
        coverage = nan;
        return;
    end

    C_method = objective_to_cost(Y_method);
    C_ref = objective_to_cost(Y_ref);

    C_all = [C_method; C_ref];

    C_method_n = normalize_cost_with_ref(C_method, C_all);
    C_ref_n = normalize_cost_with_ref(C_ref, C_all);

    n_ref = size(C_ref_n, 1);
    covered = false(n_ref, 1);

    for i = 1:n_ref

        r = C_ref_n(i, :);

        for j = 1:size(C_method_n, 1)

            p = C_method_n(j, :);

            if all(p <= r + 1e-8)
                covered(i) = true;
                break;
            end

        end

    end

    coverage = mean(covered);

end

function spread_value = compute_spread(Y_method)

    if isempty(Y_method)
        spread_value = nan;
        return;
    end

    C = objective_to_cost(Y_method);

    n = size(C, 1);

    if n <= 2
        spread_value = 0;
        return;
    end

    Cn = normalize_cost_with_ref(C, C);

    nn_dist = zeros(n, 1);

    for i = 1:n

        diff = Cn - Cn(i, :);
        dist = sqrt(sum(diff.^2, 2));
        dist(i) = inf;

        nn_dist(i) = min(dist);

    end

    spread_value = std(nn_dist);

end

function stability = compute_repeated_run_stability(metric_matrix)

    if isempty(metric_matrix) || size(metric_matrix, 1) <= 1
        stability = nan;
        return;
    end

    Z = normalize_matrix(metric_matrix);

    metric_std = std(Z, 0, 1);

    stability = exp(-mean(metric_std));

end

function Z = normalize_matrix(X)

    x_min = min(X, [], 1);
    x_max = max(X, [], 1);

    Z = (X - x_min) ./ (x_max - x_min + eps);

end

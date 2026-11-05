function S = read_synced_ecg_acc(fname, t_range, time_ref)
%READ_SYNCED_ECG_ACC  Read BI-Vital ECG + ACC synchronised to the EEG clock
%   (e.g. H039_synced_signals.csv) and plot z-scored ECG (ch overlaid on cg)
%   and z-scored |ACC| (ch overlaid on cg) to check synchronisation.
%
%   S = read_synced_ecg_acc(fname)
%   S = read_synced_ecg_acc(fname, t_range)
%   S = read_synced_ecg_acc(fname, t_range, time_ref)
%
%   fname    : CSV with columns eeg_t, ecg_aligned_ch, acc_x_ch, acc_y_ch,
%              acc_z_ch, ecg_aligned_cg, acc_x_cg, acc_y_cg, acc_z_cg
%   t_range  : [t_from t_to] in seconds, window to plot (default [0 600] = first 10 min)
%   time_ref : 'start' (default) - t = 0 at the first sample where all four
%                                  signals (ECG/ACC of ch and cg) are present
%              'eeg'             - t = eeg_t (EEG clock; 0 = EEG recording start)
%
%   S (struct):
%     fs          sampling frequency [Hz]
%     eeg_t       time on the EEG clock [s]  (n x 1)
%     t           time used for plotting, according to time_ref [s]
%     t0_eeg      eeg_t value corresponding to t = 0 for time_ref = 'start'
%     ecg_ch, ecg_cg            ECG (BI-Vital ADC counts)
%     acc_ch, acc_cg            [n x 3] ACC x, y, z
%     acc_mag_ch, acc_mag_cg    |ACC| = sqrt(x^2 + y^2 + z^2)
%   Missing samples (outside a device's recording / gaps) are NaN.
%
%   Compatible with MATLAB R2019a+.

if nargin < 2 || isempty(t_range),  t_range  = [0 600]; end
if nargin < 3 || isempty(time_ref), time_ref = 'start'; end
fname = char(fname);
if ~isfile(fname), error('read_synced_ecg_acc:noFile', 'File not found: %s', fname); end

% ---------------------------------------------------------------- read
fid = fopen(fname, 'r');
hdr = strtrim(strsplit(fgetl(fid), ','));
fclose(fid);
M = readmatrix(fname, 'NumHeaderLines', 1, 'Delimiter', ',');   % empty -> NaN
col = @(name) M(:, find(strcmp(hdr, name), 1));
for c = {'eeg_t', 'ecg_aligned_ch', 'acc_x_ch', 'ecg_aligned_cg', 'acc_x_cg'}
    if ~any(strcmp(hdr, c{1})), error('read_synced_ecg_acc:col', 'Missing column %s', c{1}); end
end

% eeg_t is stored rounded to 10 ms, but the grid is regular -> rebuild it
t_raw = col('eeg_t');
n     = numel(t_raw);
p     = polyfit((0:n-1)', t_raw, 1);
S.fs    = 1 / p(1);
S.eeg_t = p(2) + (0:n-1)' * p(1);

S.ecg_ch = col('ecg_aligned_ch');
S.ecg_cg = col('ecg_aligned_cg');
S.acc_ch = [col('acc_x_ch'), col('acc_y_ch'), col('acc_z_ch')];
S.acc_cg = [col('acc_x_cg'), col('acc_y_cg'), col('acc_z_cg')];
S.acc_mag_ch = sqrt(sum(S.acc_ch .^ 2, 2));
S.acc_mag_cg = sqrt(sum(S.acc_cg .^ 2, 2));

% ---------------------------------------------------------------- time axis
all_ok = ~isnan(S.ecg_ch) & ~isnan(S.ecg_cg) & ~isnan(S.acc_mag_ch) & ~isnan(S.acc_mag_cg);
i0 = find(all_ok, 1);
if isempty(i0), i0 = 1; end
S.t0_eeg = S.eeg_t(i0);
switch lower(time_ref)
    case 'start', S.t = S.eeg_t - S.t0_eeg; xlab = 'Time (s)  (0 = start of common ECG/ACC data)';
    case 'eeg',   S.t = S.eeg_t;            xlab = 'EEG time (s)';
    otherwise, error('read_synced_ecg_acc:ref', 'time_ref must be ''start'' or ''eeg''');
end
S.time_ref = lower(time_ref);
S.file     = fname;

% ---------------------------------------------------------------- plot
idx = S.t >= t_range(1) & S.t <= t_range(2);
if ~any(idx)
    warning('read_synced_ecg_acc:empty', 'No samples in [%g %g] s (data span %.1f..%.1f s).', ...
        t_range(1), t_range(2), S.t(1), S.t(end));
    return
end
tt = S.t(idx);
[~, base] = fileparts(fname);

% z-score within the plotted window (NaN-safe); ch overlaid on cg
c_cg = [0.80 0.15 0.15];
c_ch = [0.12 0.31 0.55];
pairs = { zscore_nan(S.ecg_cg(idx)),     zscore_nan(S.ecg_ch(idx)),     'ECG (z)'
          zscore_nan(S.acc_mag_cg(idx)), zscore_nan(S.acc_mag_ch(idx)), '|ACC| (z)' };

figure('Color', 'w', 'Position', [80 80 1400 750]);
ax = gobjects(2, 1);
for k = 1:2
    ax(k) = subplot(2, 1, k);
    hold(ax(k), 'on');
    plot(ax(k), tt, pairs{k, 1}, 'Color', c_cg, 'LineWidth', 0.8, 'DisplayName', 'cg');
    plot(ax(k), tt, pairs{k, 2}, 'Color', c_ch, 'LineWidth', 0.6, 'DisplayName', 'ch');
    hold(ax(k), 'off');
    ylabel(ax(k), pairs{k, 3});
    legend(ax(k), 'Location', 'northeast');
    grid(ax(k), 'on'); box(ax(k), 'on');
end
set(ax(1), 'XTickLabel', []);
linkaxes(ax, 'x');
xlim(ax(1), [tt(1) tt(end)]);
xlabel(ax(2), xlab);
title(ax(1), sprintf('%s | %.0f-%.0f s | fs = %.1f Hz | z-scored in window', ...
    base, t_range(1), t_range(2), S.fs), 'Interpreter', 'none');
end


function z = zscore_nan(x)
ok = ~isnan(x);
if nnz(ok) < 2, z = x; return; end
s = std(x(ok));
if s == 0, s = 1; end
z = (x - mean(x(ok))) / s;
end

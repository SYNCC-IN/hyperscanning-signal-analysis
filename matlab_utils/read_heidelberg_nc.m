function [signals, time, meta] = read_heidelberg_nc(fname, do_plot)
%READ_HEIDELBERG_NC  Read a SYNCC-IN by-task NetCDF export (Heidelberg / Warsaw).
%
%   [signals, time, meta] = read_heidelberg_nc(fname)
%   [signals, time, meta] = read_heidelberg_nc(fname, do_plot)
%
%   Files written by export_heidelberg_by_task (src/export.py), e.g.
%   H032_EEG_ch_passive_movies.nc, H032_IBI_cg_talk.nc.
%
%   Outputs
%     signals : [n_time x n_channels] double
%     time    : [n_time x 1] double, seconds; 0 = start of the first event
%               of the task chunk, negative / > task_duration = margins
%     meta    : struct with ALL attributes of the 'signals' variable.
%               Numeric strings are converted to numbers, JSON attributes
%               (*_json, task_events_structure, ...) are decoded to
%               structs / cells. Convenience fields added:
%                 meta.channel_names  - cellstr {1 x n_channels}
%                 meta.events         - struct array: name, start_s (rel.), duration_s, end_s
%                 meta.file           - source file path
%
%   do_plot (default true): plot signal vs time with events highlighted,
%   margins shaded grey; multi-channel data (EEG) stacked vertically.
%
%   Compatible with MATLAB R2019b+.

if nargin < 2 || isempty(do_plot), do_plot = true; end
fname = char(fname);
if ~isfile(fname), error('read_heidelberg_nc:noFile', 'File not found: %s', fname); end

% ---------------------------------------------------------------- data
time    = double(ncread(fname, 'time'));
time    = time(:);
signals = double(ncread(fname, 'signals'));   % MATLAB returns [channel x time]
if size(signals, 1) ~= numel(time)
    signals = signals.';
end

% ---------------------------------------------------------------- metadata
info = ncinfo(fname, 'signals');
meta = struct();
for k = 1:numel(info.Attributes)
    name = info.Attributes(k).Name;
    if strcmp(name, '_FillValue'), continue; end
    meta.(matlab.lang.makeValidName(name)) = parse_attr(info.Attributes(k).Value);
end
meta.file = fname;

% channel names: csv attr -> json attr -> 'channel' coordinate variable
ch = {};
if isfield(meta, 'channel_names_csv') && ischar(meta.channel_names_csv) && ~isempty(meta.channel_names_csv)
    ch = strtrim(strsplit(meta.channel_names_csv, ','));
elseif isfield(meta, 'channel_names_json')
    ch = to_cellstr(meta.channel_names_json);
end
if numel(ch) ~= size(signals, 2) || all(cellfun(@isempty, ch))
    try
        c  = ncread(fname, 'channel');            % char [strlen x n_ch]
        ch = strtrim(cellstr(c.'));
    catch
    end
end
mod_name = get_field(meta, 'modality', 'signal');
for i = 1:size(signals, 2)                       % e.g. IBI / RMSSD: empty name
    if i > numel(ch) || isempty(ch{i})
        if size(signals, 2) == 1, ch{i} = mod_name; else, ch{i} = sprintf('ch%d', i); end
    end
end
meta.channel_names = ch(1:size(signals, 2));

% events (relative to chunk time axis)
ev = struct('name', {}, 'start_s', {}, 'duration_s', {}, 'end_s', {});
if isfield(meta, 'task_events_structure')
    raw = meta.task_events_structure;
    if isstruct(raw), raw = num2cell(raw); end
    for i = 1:numel(raw)
        e = raw{i};
        if isfield(e, 'start_rel_s'), t0 = e.start_rel_s; else, t0 = e.start_s; end
        ev(end+1).name     = char(e.name);          %#ok<AGROW>
        ev(end).start_s    = double(t0);
        ev(end).duration_s = double(e.duration_s);
        ev(end).end_s      = ev(end).start_s + ev(end).duration_s;
    end
end
meta.events = ev;

if do_plot
    plot_nc_signals(signals, time, meta);
end
end


% ======================================================================
function plot_nc_signals(signals, time, meta)
n_ch     = size(signals, 2);
t_first  = time(1);
t_last   = time(end);
task_dur = get_field(meta, 'task_duration', NaN);
units    = get_field(meta, 'units', '');

figure('Color', 'w', 'Position', [80 80 1400 800]);
ax = axes; hold(ax, 'on');

if n_ch > 1      % ---- stacked (EEG): normalise by median channel SD
    sd = zeros(1, n_ch);
    for i = 1:n_ch
        x = signals(~isnan(signals(:, i)), i);
        if numel(x) > 1, sd(i) = std(x); end
    end
    sc = median(sd(sd > 0));
    if isempty(sc) || isnan(sc), sc = 1; end
    spacing = 8;
    offsets = (n_ch - 1:-1:0) * spacing;      % first channel on top
    Y       = signals ./ sc + offsets;
    y_lim   = [-spacing, (n_ch - 1) * spacing + spacing];
else             % ---- single channel (ECG / IBI / RMSSD): absolute units
    Y     = signals;
    lo    = min(Y); hi = max(Y);                % NaNs ignored
    if isempty(lo) || isnan(lo) || lo == hi, lo = lo - 1; hi = hi + 1; end
    pad   = 0.15 * (hi - lo);
    y_lim = [lo - pad, hi + 2 * pad];         % headroom for event labels
end

% 1) margins (grey)
gray = [0.80 0.80 0.80];
if t_first < 0
    shade(ax, [t_first 0], y_lim, gray, 0.6);
end
if ~isnan(task_dur) && t_last > task_dur
    shade(ax, [task_dur t_last], y_lim, gray, 0.6);
end

% 2) events (coloured, labelled)
ev   = meta.events;
cols = lines(max(numel(ev), 1));
for i = 1:numel(ev)
    shade(ax, [ev(i).start_s ev(i).end_s], y_lim, cols(i, :), 0.18);
    text(ax, ev(i).start_s + 0.5 * ev(i).duration_s, y_lim(2), ev(i).name, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontWeight', 'bold', 'Color', 0.6 * cols(i, :), 'Interpreter', 'none');
end

% 3) signals on top
plot(ax, time, Y, 'Color', [0.12 0.31 0.55], 'LineWidth', 0.6);

% 4) chunk boundaries
plot(ax, [0 0], y_lim, 'k--', 'LineWidth', 0.8);
if ~isnan(task_dur), plot(ax, [task_dur task_dur], y_lim, 'k--', 'LineWidth', 0.8); end

hold(ax, 'off');
xlim(ax, [t_first t_last]); ylim(ax, y_lim);
xlabel(ax, 'Time (s)  (0 = start of first event)');
if n_ch > 1
    set(ax, 'YTick', fliplr(offsets), 'YTickLabel', fliplr(meta.channel_names), ...
        'TickLabelInterpreter', 'none');
    ylabel(ax, sprintf('Channels (normalised, 1 unit = median SD = %.3g %s)', sc, units));
else
    ylabel(ax, sprintf('%s [%s]', meta.channel_names{1}, units), 'Interpreter', 'none');
end
title(ax, sprintf('%s | dyad %s | %s | %s | margin %gs', ...
    get_field(meta, 'modality', ''), get_field(meta, 'dyad_id', ''), ...
    get_field(meta, 'who', ''), get_field(meta, 'task_name', ''), ...
    get_field(meta, 'time_margin_s', NaN)), 'Interpreter', 'none');
box(ax, 'on');
end


% ======================================================================
function shade(ax, xr, yr, col, alpha)
patch(ax, [xr(1) xr(2) xr(2) xr(1)], [yr(1) yr(1) yr(2) yr(2)], col, ...
    'FaceAlpha', alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function v = parse_attr(v)
% Convert NetCDF attribute value: JSON -> struct/cell, numeric string -> double.
if isstring(v), v = char(v); end
if ~ischar(v), v = double(v); return; end
s = strtrim(v);
if isempty(s), v = s; return; end
if s(1) == '{' || s(1) == '['
    % Python json.dumps may write NaN / Infinity, which jsondecode rejects
    js = regexprep(s, '(?<![\w"])(-?Infinity|NaN)(?![\w"])', 'null');
    try
        v = jsondecode(js);
        return
    catch
    end
end
num = str2double(s);
if ~isnan(num) || strcmpi(s, 'nan')
    v = num;
else
    v = s;
end
end

function c = to_cellstr(x)
if iscell(x), c = cellfun(@char, x(:).', 'UniformOutput', false);
elseif isstring(x), c = cellstr(x(:).');
elseif ischar(x), c = {x};
else, c = {};
end
end

function v = get_field(s, f, default)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = default; end
if isnumeric(v) && ischar(default), v = num2str(v); end
end

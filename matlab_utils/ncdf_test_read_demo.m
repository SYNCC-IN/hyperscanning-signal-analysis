fname = 'W_000_EEG_ch_Brave.nc';

% Channel names read order (MATLAB R2019b compatible):
% 1) signals:channel_names_csv
% 2) signals:channel_names_json
% 3) fallback to legacy coordinate variable: channel

% --- Read attributes from 'signals' variable ---
dyad_id        = ncreadatt(fname, 'signals', 'dyad_id');
who            = ncreadatt(fname, 'signals', 'who');
sampling_freq  = ncreadatt(fname, 'signals', 'sampling_freq');
event_name     = ncreadatt(fname, 'signals', 'event_name');
event_start    = ncreadatt(fname, 'signals', 'event_start');
event_duration = ncreadatt(fname, 'signals', 'event_duration');
time_margin_s  = ncreadatt(fname, 'signals', 'time_margin_s');

% --- Decode metadata JSON ---
metadata_json_str = ncreadatt(fname, 'signals', 'metadata_json');
metadata          = jsondecode(metadata_json_str);

% --- Display basic info ---
fprintf('Dyad ID:        %s\n', dyad_id);
fprintf('Member:         %s\n', who);
fprintf('Sampling freq:  %.1f Hz\n', sampling_freq);
fprintf('Event:          %s\n', event_name);
fprintf('Event start:    %.2f s\n', event_start);
fprintf('Event duration: %.2f s\n', event_duration);
fprintf('Time margin:    %.2f s\n', time_margin_s);

% --- Display child info from metadata ---
fprintf('\n--- Child Info ---\n');
fprintf('Age (months):   %.1f\n', metadata.child_info.age_months);
fprintf('Group:          %s\n',   metadata.child_info.group);
fprintf('Sex:            %s\n',   metadata.child_info.sex);

% --- Display movie (event) order ---
% event_order lists Peppa / Incredibles / Brave sorted by their start time
fprintf('\n--- Movie Order ---\n');
if isfield(metadata, 'event_order') && ~isempty(metadata.event_order)
    event_order = metadata.event_order;
    % jsondecode returns a cell array of strings
    for k = 1:numel(event_order)
        fprintf('  %d. %s\n', k, event_order{k});
    end
else
    fprintf('  (event_order not available in this file)\n');
end

% --- Display EEG processing info ---
fprintf('\n--- EEG Processing ---\n');
fprintf('Reference:         %s\n',   metadata.eeg.references);
fprintf('Notch filter:      %.0f Hz\n', metadata.eeg.filtration.notch.freq);
fprintf('Notch applied:     %d\n',   metadata.eeg.filtration.notch.applied);
fprintf('Low-pass applied:  %d\n',   metadata.eeg.filtration.low_pass.applied);
fprintf('High-pass applied: %d\n',   metadata.eeg.filtration.high_pass.applied);

% --- Read time ---
time = ncread(fname, 'time');

% --- Read channel names (MATLAB 2019b-friendly) ---
channel_names = {};

% Preferred: CSV attribute written on 'signals'
try
    channel_names_csv = ncreadatt(fname, 'signals', 'channel_names_csv');
    if ischar(channel_names_csv) || isstring(channel_names_csv)
        channel_names = strtrim(strsplit(char(channel_names_csv), ','));
    end
catch
end

% Alternative: JSON attribute written on 'signals'
if isempty(channel_names)
    try
        channel_names_json = ncreadatt(fname, 'signals', 'channel_names_json');
        decoded_names = jsondecode(char(channel_names_json));
        if isstring(decoded_names)
            channel_names = cellstr(decoded_names(:));
        elseif iscell(decoded_names)
            channel_names = decoded_names;
        end
    catch
    end
end

% Fallback for older files: char matrix from coordinate variable 'channel'
if isempty(channel_names)
    ncid          = netcdf.open(fname, 'NC_NOWRITE');
    varid         = netcdf.inqVarID(ncid, 'channel');
    channel       = netcdf.getVar(ncid, varid);
    channel_names = strtrim(cellstr(channel.'));
    netcdf.close(ncid);
end

% --- Read signals ---
signals = ncread(fname, 'signals');

% Transpose if needed so signals is [n_time x n_channels]
if size(signals, 1) ~= length(time)
    signals = signals.';
end

fprintf('\nSignals size: %d samples x %d channels\n', size(signals,1), size(signals,2));
fprintf('Duration:     %.2f s at %.0f Hz\n', length(time)/sampling_freq, sampling_freq);

% --- Plot all channels with offset ---
n_plot        = size(signals, 2);
spacing       = 5 * std(signals(:));
event_end_rel = event_duration;
t_start       = time(1);
t_end         = time(end);
y_min         = -spacing;
y_max         = (n_plot - 1) * spacing + spacing;

figure;
hold on;

% 1. Draw shading FIRST (behind signals)
if t_start < 0
    patch([t_start 0 0 t_start], [y_min y_min y_max y_max], ...
        [0.85 0.85 0.85], ...
        'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
end

if t_end > event_end_rel
    patch([event_end_rel t_end t_end event_end_rel], [y_min y_min y_max y_max], ...
        [0.85 0.85 0.85], ...
        'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
end

% 2. Draw signals ON TOP
for i = 1:n_plot
    offset = (i - 1) * spacing;
    plot(time, signals(:, i) + offset, 'DisplayName', channel_names{i});
end

% 3. Event boundary lines
xline(0,             'k--', 'Event start', 'LabelVerticalAlignment', 'bottom');
xline(event_end_rel, 'k--', 'Event end',   'LabelVerticalAlignment', 'bottom');

hold off;

xlabel('Time (s)')
ylabel('Channels')
title(sprintf('EEG — Dyad %s | %s | Event: %s', dyad_id, who, event_name))
legend('Location', 'bestoutside')
yticks(0 : spacing : (n_plot-1) * spacing)
yticklabels(channel_names(1:n_plot))
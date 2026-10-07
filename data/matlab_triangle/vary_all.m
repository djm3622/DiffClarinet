clear; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

Fs = 22025;
T = 40000;

L_values = 100:5:200;
a_values = 0.1:0.1:0.9;
delay_gain_values = linspace(0.7, 0.999, 18);
pluck_position_ratios = [0.1, 0.3, 0.5];

impulse_gain = 1.0;
headroom = 0.95;

output_dir = fullfile(fileparts(script_dir), 'vary_all_triangle');
if ~isfolder(output_dir)
    mkdir(output_dir);
end

% The Python loader validates this contract and removes exactly one sample.
manifest_path = fullfile(output_dir, 'manifest.json');
manifest_file = fopen(manifest_path, 'w');
if manifest_file < 0
    error('Could not write triangle dataset manifest.');
end
fprintf(manifest_file, ['{"sample_rate": %d, "causal_samples": %d, ' ...
    '"leading_zero": 1, "sign": "negative", ' ...
    '"audio_format": "24-bit PCM WAV"}\n'], Fs, T);
fclose(manifest_file);

for i = 1:numel(delay_gain_values)
    delay_gain = delay_gain_values(i);

    for j = 1:numel(a_values)
        a = a_values(j);

        for k = 1:numel(L_values)
            L = L_values(k);

            for p = 1:numel(pluck_position_ratios)
                pluck_position_ratio = pluck_position_ratios(p);
                A = round(pluck_position_ratio * L);

                [y, noise, scale, peak_raw] = plucked_gen( ...
                    T, L, impulse_gain, delay_gain, a, A, headroom ...
                );

                file_stem = sprintf( ...
                    'vary_all_triangle_delay_gain_%.3f_a_%.1f_L_%03d_pos_%.1f_A_%03d', ...
                    delay_gain, a, L, pluck_position_ratio, A ...
                );

                wav_path = fullfile(output_dir, [file_stem, '.wav']);
                mat_path = fullfile(output_dir, [file_stem, '.mat']);

                audiowrite(wav_path, y, Fs, 'BitsPerSample', 24);
                save(mat_path, ...
                    'noise', 'scale', 'peak_raw', ...
                    'A', 'pluck_position_ratio', 'impulse_gain', ...
                    'L', 'a', 'delay_gain', 'Fs', 'T');
            end
        end
    end
end

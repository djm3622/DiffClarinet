function generate_uniform_grid(generator_root, output_root)
% Uniform-grid comb and triangle targets for the delay benchmark.
%
% L is uniform on the integers 100..200 and p is uniform on 1..floor(L/2) given
% L, which keeps p within the model's position range and avoids the triangle's
% p vs L-p mirror ambiguity. K and a are uniform and continuous, rounded to six
% decimals so that the filename labels equal the synthesized values.
%
% generator_root is the plucked_string folder holding pluck_position and
% pluck_triangle, whose plucked_gen functions synthesize the targets.
% output_root is the repository's data folder.

    seed = 20261105;
    count = 160;

    rng(seed, 'twister');
    comb = draw_parameters(count);
    triangle = draw_parameters(count);

    addpath(fullfile(generator_root, 'pluck_position'));
    write_comb(comb, fullfile(output_root, 'uniform_grid_pluck'));
    rmpath(fullfile(generator_root, 'pluck_position'));

    addpath(fullfile(generator_root, 'pluck_triangle'));
    write_triangle(triangle, fullfile(output_root, 'uniform_grid_triangle'));
    rmpath(fullfile(generator_root, 'pluck_triangle'));
end


function parameters = draw_parameters(count)
    L = randi([100, 200], count, 1);
    p = arrayfun(@(length) randi([1, floor(length / 2)]), L);
    K = round(0.700 + 0.299 * rand(count, 1), 6);
    a = round(0.1 + 0.8 * rand(count, 1), 6);
    parameters = table(L, p, K, a);
end


function write_comb(parameters, output_dir)
    Fs = 22025;
    T = 40000;
    impulse_gain = 1.0;
    headroom = 0.95;
    excitation_length = 200;

    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    for i = 1:height(parameters)
        L = parameters.L(i);
        pluck_delay = parameters.p(i);
        delay_gain = parameters.K(i);
        a = parameters.a(i);
        pluck_position_ratio = pluck_delay / L;

        [y, noise, scale, peak_raw] = plucked_gen( ...
            T, L, impulse_gain, delay_gain, a, ...
            pluck_delay, headroom, excitation_length ...
        );

        % MatlabPluckData accepts only unscaled comb targets.
        if scale ~= 1
            error('Scaled comb target at row %d.', i);
        end

        file_stem = sprintf( ...
            "uniform_grid_pluck_delay_gain_%.6f_a_%.6f_L_%03d_pos_%.4f_d_%03d", ...
            delay_gain, a, L, pluck_position_ratio, pluck_delay ...
        );

        audiowrite(fullfile(output_dir, file_stem + ".wav"), y, Fs);

        save( ...
            fullfile(output_dir, file_stem + ".mat"), ...
            "noise", "scale", "peak_raw", ...
            "pluck_delay", "pluck_position_ratio", "impulse_gain" ...
        );
    end
end


function write_triangle(parameters, output_dir)
    Fs = 22025;
    T = 40000;
    impulse_gain = 1.0;
    headroom = 0.95;

    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    % The Python loader validates this contract and removes exactly one sample.
    manifest_file = fopen(fullfile(output_dir, 'manifest.json'), 'w');
    if manifest_file < 0
        error('Could not write triangle dataset manifest.');
    end
    fprintf(manifest_file, ['{"sample_rate": %d, "causal_samples": %d, ' ...
        '"leading_zero": 1, "sign": "negative", ' ...
        '"audio_format": "24-bit PCM WAV"}\n'], Fs, T);
    fclose(manifest_file);

    for i = 1:height(parameters)
        L = parameters.L(i);
        A = parameters.p(i);
        delay_gain = parameters.K(i);
        a = parameters.a(i);
        pluck_position_ratio = A / L;

        [y, noise, scale, peak_raw] = plucked_gen( ...
            T, L, impulse_gain, delay_gain, a, A, headroom ...
        );

        file_stem = sprintf( ...
            'uniform_grid_triangle_delay_gain_%.6f_a_%.6f_L_%03d_pos_%.4f_A_%03d', ...
            delay_gain, a, L, pluck_position_ratio, A ...
        );

        audiowrite(fullfile(output_dir, [file_stem, '.wav']), y, Fs, ...
            'BitsPerSample', 24);
        save(fullfile(output_dir, [file_stem, '.mat']), ...
            'noise', 'scale', 'peak_raw', ...
            'A', 'pluck_position_ratio', 'impulse_gain', ...
            'L', 'a', 'delay_gain', 'Fs', 'T');
    end
end

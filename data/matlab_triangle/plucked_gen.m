function [Y, noise, scale, peak_raw] = plucked_gen( ...
    T, L, impulse_gain, delay_gain, a, A, headroom ...
)
    % Triangle: 0 at sample 0, 1 at A, 0 at L, then zero.
    validateattributes(T, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(L, {'numeric'}, {'scalar', 'integer', '>=', 2});
    validateattributes(A, {'numeric'}, {'scalar', 'integer', '>=', 1, '<', L});
    validateattributes(headroom, {'numeric'}, {'scalar', '>', 0, '<=', 1});

    if T <= L
        error('T must exceed L.');
    end

    n = 0:L;
    triangle = min(n / A, (L - n) / (L - A));

    excitation = zeros(1, T);
    excitation(1:L+1) = ...
        -impulse_gain * delay_gain * triangle;

    % Same allpass filter and positive-feedback loop as pluck_position.
    b = [a / 2, (a + 1) / 2, 1 / 2];
    denominator = zeros(1, L + 3);
    denominator(1) = 1;
    denominator(2) = a;
    denominator(L+1:L+3) = -delay_gain * b;

    causal_output = filter(b, denominator, excitation);

    % Match the existing generator's leading-zero convention.
    Y = [0; causal_output(:)];

    peak_raw = max(abs(Y));
    scale = min(1.0, headroom / max(peak_raw, eps));
    Y = Y * scale;

    % Save the actual input signal used to produce the scaled WAV.
    noise = excitation(1:L+1) * scale;
end
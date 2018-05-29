% Deduce how to convert a cubic spline to its bezier representation.
% Author: Chi Zhang (chizhang@deepmotion.ai)

syms a b c d;   % Coefficients of a cubic spline segment.
syms A B C D;   % Coefficients of the corresponding cubic bezier.
syms s t t0 dt;

% s = (t - t0) / dt;
% X = A*((t-t0)/dt)^3 + B*((t-t0)/dt)^2 + C*(t-t0)/dt + D

[A, B, C, D] = solve([ ...
    a == (A/dt^3), ...
    b == (B/dt^2 - (3*A*t0)/dt^3), ...
    c == (C/dt + (3*A*t0^2)/dt^3 - (2*B*t0)/dt^2), ...
    d == D - (A*t0^3)/dt^3 + (B*t0^2)/dt^2 - (C*t0)/dt ...
    ], [A, B, C, D])
     
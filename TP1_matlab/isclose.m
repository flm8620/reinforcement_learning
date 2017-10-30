function flag = isclose(a,b,atol,rtol)
if nargin < 4
    rtol = 1e-5;
end
if nargin < 3
    atol = 1e-8;
end
flag = abs(a - b) <= (atol + rtol * abs(b));
end
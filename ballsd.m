%% BALLSD
%
% Optimization using the Bayesian adaptive locally linear stochastic
% descent algorithm. In general the syntax is the same as Matlab's other
% optimization functions, including fminsearch (simplex), lsqnonlin
% (trust-region-reflective or Levenberg-Marquardt), simulannealbnd
% (simulated annealing), and ga (genetic algorithm). Specifically:
%
% X = ballsd(FUN,X0) starts at X0 and attempts to find a local
% minimizer X of the function FUN. FUN is a function handle. FUN accepts
% input X and returns a scalar function value F evaluated at X. X0 can be a
% scalar, vector, or array of any size.
% 
% X = ballsd(FUN,X0,OPTIONS) minimizes with the default optimization
% parameters replaced by values in the structure OPTIONS. Fields {defaults}
% are:
%       stepsize {0.2} -- Initial step size as a fraction of each parameter
%           sinc {2} -- Step size learning rate (increase)
%           sdec {2} -- Step size learning rate (decrease)
%           pinc {2} -- Parameter selection learning rate (increase)
%           pdec {2} -- Parameter selection learning rate (decrease)
%       pinitial {ones(2*size(X0))} -- Set initial parameter selection probabilities
%       sinitial {[]} -- Set initial step sizes; if empty, calculated from stepsize instead
%    MaxFunEvals {1000*length(X0)} -- Maximum number of function evaluations
%        MaxIter {1e4} -- Maximum number of iterations (1 iteration = 1 function evaluation)
%         TolFun {1e-6} -- Minimum change in objective function
%           TolX {1e-6*mean(X0)} -- Minimum change in parameters
% StallIterLimit {100} -- Number of iterations over which to calculate TolFun
%   maxarraysize {1e6} -- Limit on MaxIter and StallIterLimit to ensure arrays don't get too big
% 
% [X,FVAL] = ballsd(...) returns the value of the objective function at X.
% 
% [X,FVAL,EXITFLAG] = ballsd(...) returns an EXITFLAG that describes the
% exit condition of fminsearch. Possible values of EXITFLAG and the
% corresponding exit conditions are:
%      0 -- Maximum number of function evaluations or iterations reached.
%      1 -- Improvement in objective function below minimum threshold.
%      2 -- Step size below threshold.
%     -1 -- Algorithm terminated for other reasons.
% 
% [X,FVAL,EXITFLAG,OUTPUT] = ballsd(...) returns a structure OUTPUT with
% the following fields:
%   iterations -- Number of iterations
%    funcCount -- Number of function evaluations
%         fval -- Value of objective function at each iteration
%            x -- Vector of parameter values at each iteration
% 
%
% Example:
%
% BALLSD can be used in the same way as fminsearch, e.g.
%
%     [x, fval, exitflag, output] = ballsd(@(x) norm(x), [1 2 3]);
%
%     fval =
%        4.4409e-16
%     output.funcCount =
%        102
%
% In this example BALLSD finds a vastly better solution than fminsearch
% and in considerably fewer iterations:
%
%     [x, fval, exitflag, output] = fminsearch(@(x) norm(x), [1 2 3]);
%
%     fval =
%        5.2741e-05
%     output.funcCount =
%        167
%
%
% Version: 2014jun05 by Cliff Kerr (cliff@thekerrlab.com)

function [x, fval, exitflag, output] = ballsd(funfcn, x, varargin)

narginchk(2,3) % Ensure the correct number of input arguments were supplied
x = x(:); % Turn it into a column vector
nparams = length(x); % Number of parameters

%% Set defaults (cf. defaults for lsqnonlin)
options.stepsize = 0.2; % Initial step size as a fraction of each parameter
options.sinc = 2; % Step size learning rate (increase)
options.sdec = 2; % Step size learning rate (decrease)
options.pinc = 2; % Parameter selection learning rate (increase)
options.pdec = 2; % Parameter selection learning rate (decrease)
options.pinitial = ones(2*nparams,1); % Set initial parameter selection probabilities -- uniform by default
options.sinitial = []; % Set initial step sizes -- if empty, calculated from options.stepsize instead
options.MaxFunEvals = 1000*nparams; % Maximum number of function evaluations
options.MaxIter = 1e4; % Maximum number of iterations (1 iteration = 2 function evaluations)
options.TolFun = 1e-6; % Minimum change in objective function
options.TolX = 1e-6*mean(x); % Minimum change in parameters
options.StallIterLimit = 100; % Number of iterations over which average change in fitness function value at current point is less than options.TolFun
options.maxarraysize = 1e6; % To stop over-ambitious users, set a hard upper limit on array sizes

%% Handle custom options
if nargin == 3 % Check that the argument was supplied
    optionfields = fieldnames(options);
    if isstruct(varargin{1}) % Check that it's a structure
        for f = 1:length(optionfields) % Loop over all options fields
            fld = optionfields{f};
            if isfield(varargin{1},fld) % Check if this field exists in the input structure
                options.(fld) = varargin{1}.(fld); % If so, reset the option
            end
        end
    else
        error('Third argument must be the options structure')
    end
end
options.StallIterLimit = min(options.StallIterLimit, options.maxarraysize); % Don't by default let users create arrays larger than this -- slow and pointless
options.MaxIter = min(options.MaxIter, options.maxarraysize);


%% Initialization
funfcn = fcnchk(funfcn); % Uhh...do this, not sure why, but Matlab's other fitting algorithms do it
p = options.pinitial; % Initialize the probabilities
if isempty(options.sinitial), s = abs(options.stepsize*x); s = [s;s]; % Set initial step sizes; need to duplicate since two for each parameter
else s = abs(options.sinitial); % Ensure always positive
end
s(s==0) = mean(s(s~=0)); % Replace step sizes of zeros with the mean of non-zero entries
fval = funfcn(x); % Calculate initial value of the objective function
count = 0; % Keep track of how many iterations have occurred
exitflag = -1; % Set default exit flag
errorhistory = zeros(options.StallIterLimit,1); % Store previous error changes -- but not more than a million, that's ridiculous!
if nargout==4 % Include additional output structure
    output.fval = zeros(options.MaxIter,1); % Store all objective function values
    output.x = zeros(options.MaxIter,nparams); % Store all parameters
end

%% Loop
while 1
    
    % Calculate next step
    count = count+1; % On each iteration there are two function evaluations
    p = p/sum(p); % Normalize probabilities
    cumprobs = cumsum(p); % Calculate the cumulative distribution
    choice = find(cumprobs > rand(), 1); % Choose a parameter and upper/lower at random
    par = mod(choice-1,nparams)+1; % Which parameter was chosen
    pm = floor((choice-1)/nparams); % Plus or minus
    xnew = x; % Initialize the new parameter set
    xnew(par) = xnew(par) + ((-1)^pm)*s(choice); % Calculate the new parameter set
    fvalnew = funfcn(xnew); % Calculate the objective function for the new parameter set
    errorhistory(mod(count,options.StallIterLimit)+1) = fval - fvalnew; % Keep track of improvements in the error
    
    % Check if this step was an improvement
    if fvalnew < fval % New parameter set is better than previous one
        p(choice) = p(choice)*options.pinc; % Increase probability of picking this parameter again
        s(choice) = s(choice)*options.sinc; % Increase size of step for next time
        x = xnew; % Reset current parameters
        fval = fvalnew; % Reset current error
    elseif fvalnew >= fval % New parameter set is the same or worse than the previous one
        p(choice) = p(choice)/options.pdec; % Increase probability of picking this parameter again
        s(choice) = s(choice)/options.sdec; % Increase size of step for next time
    end
    
    % Optionally store output information
    if nargout==4 % Include additional output structure
        output.fval(count) = fval; % Store objective function evaluations
        output.x(count,:) = x; % Store parameters
    end
    
    % Stopping criteria
    if count+1 >= options.MaxFunEvals, exitflag = 0; break, end % Stop if the function evaluation limit is exceeded
    if count >= options.MaxIter, exitflag = 0; break, end % Stop if the iteration limit is exceeded
    if mean(s) < options.TolX, exitflag = 1; break, end % Stop if the step sizes are too small
    if (count > options.StallIterLimit) && (mean(errorhistory) < options.TolFun), exitflag = 2; break, end % Stop if improvement is too small    

end

% Store additional output if required
if nargout==4 % Include additional output structure
    output.iterations = count; % Number of iterations
    output.funcCount = count+1; % Number of function evaluations
    output.fval = output.fval(1:count); % Function evaluations
    output.x = output.x(1:count,:); % Parameters
end

end

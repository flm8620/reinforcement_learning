clc;
clear all;
close all;
grid1 = {
    {'','','',1};
    {'','x','',-1};
    {'','','',''}
    };

env = GridWorld(grid1, 0.95);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% investigate the structure of the environment
% - env.n_states: the number of states
% - env.state2coord (cell): converts state number to coordinates (row, col)
% - env.coord2state (matrix): converts coordinates (row, col) into state number
% - env.action_names (cell): converts action number [1,4] into a named action
% - env.state_actions (cell): for each state stores the action availables
%   For example
%       env.state_actions{5} -> [2,4]
%       env.action_names{env.state_actions{5}} -> ['down', 'up']
% - env.gamma: discount factor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
env.state2coord
env.coord2state
env.state_actions{5}
%%
fprintf('\nActions per state\n\n');
for i = 1:length(env.state_actions)
  fprintf('s%2d : [', i);
  for j = 1:length(env.state_actions{i})
    fprintf('%s ', env.action_names{env.state_actions{i}(j)});
  end
  fprintf(']\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Policy definition
% If you want to represent deterministic action you can just use the number of
% the action. Recall that in the terminal states only action 0 (right) is
% defined.
% In this case, you can use render_policy to visualize the policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pol = {2, 3, 1, 1, 2, 2, 1, 1, 1, 1, 4};
render_policy(env, pol)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Try to simulate a trajectory
% you can use env.step(s,a, render=True) to visualize the transition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
env.render = 1;
state = 1;
fps = 10;
figure();
hold on;
for i = 1:50
    n_actions = length(env.state_actions{state});
    action = randi([1,n_actions]);
    action = env.state_actions{state}(action);
    [nexts, reward, term] = env.step(state,action);
    state = nexts;
    pause(1./fps)
end
hold off;
env.render = 0;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can also visualize the q-function using render_q
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first get the maximum number of actions available
max_act = max(cellfun(@(c) length(c), env.state_actions));
q = rand(env.n_states, max_act);
render_q(env, q)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Work to do: Q4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here the v-function and q-function to be used for question 4 (consider mainly
% the v-function)
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, ...
    0.67106071, -0.99447514, 0.00000000, -0.82847001, ...
    -0.87691855, -0.93358351, -0.99447514];
q_q4 = {{0.87691855, 0.65706417},...
    {0.92820033, 0.84364237},...
    {0.98817903, -0.75639924, 0.89361129},...
    {0.00000000},...
    {-0.62503460, 0.67106071},...
    {-0.99447514, -0.70433689, 0.75620264},...
    {0.00000000},...
    {-0.82847001, 0.49505225},...
    {-0.87691855, -0.79703229},...
    {-0.93358351, -0.84424050, -0.93896668},...
    {-0.89268904, -0.99447514}
    };

N = zeros(env.n_states, 4);
Q = zeros(env.n_states, 4);
Tmax = 1/(1-env.gamma)*10;
for e = 1:5000
    x0 = env.reset();
    actions = env.state_actions{x0};
    if any(actions==1)
        a0 = 1;
    else
        a0 = 4;
    end
    N(x0,a0)=N(x0,a0)+1;
    tot = 0;
    terminated = false;
    gamma_power = 1.0;
    x=x0;
    a=a0;
    t=0;
    while t<Tmax && ~terminated
        [x, reward, terminated] = env.step(x, a);
        tot = tot + reward * gamma_power;
        gamma_power = gamma_power * env.gamma;
        actions = env.state_actions{x};
        if any(actions==1)
            a = 1;
        else
            a = 4;
        end
        t=t+1;
    end
    Q(x0,a0) = Q(x0,a0) + tot;    
end
Q = Q./N;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Work to do: Q5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, ...
    0.82369294, 0.92820033, 0.00000000, 0.77818504, ...
    0.82369294, 0.87691855, 0.82847001];

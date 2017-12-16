clc;
clear all;
close all;

random_state = randi([0, 24532523]);

noise = 0.1;
% model = ToyLinearMABModel(8, 200, random_state, noise);

model = ColdStartMovieLensModel(random_state, noise);

n_a = model.n_actions();
d = model.n_features();

T = 6000;
nb_simu = 50; % you may want to change this!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the algorithms
% - Random
% - Linear UCB
% - Eps Greedy
% and test it!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
alpha_t = 1;
lambda = 1;
alg_name = ['Linear UCB, alpha=', num2str(alpha_t), ' lambda=',num2str(lambda)];

fprintf('Simulating %s...', alg_name);

regret = zeros(nb_simu, T);
norm_dist = zeros(nb_simu, T);
Z = model.features;

h = waitbar(0,'Initializing waitbar...');
for k = 1:nb_simu
    waitbar(k/nb_simu,h,sprintf('%d/%d', k, nb_simu));
    %choice_history = zeros(T,1);
    
    An = lambda * eye(d,d);
    bn = zeros(d,1);
    theta_hat = An\bn;
    for t = 1:T
        if t==1
            a_t = randi(n_a);
        else
            beta = alpha_t * sum((Z/An).*Z,2);
            [~,a_t] = max(Z * theta_hat + beta); % algorithm picks the action
        end
        r_t = model.reward(a_t); % get the reward
        %choice_history(t)=a_t;
        % do something (update algorithm)
        theta_a = Z(a_t,:)';
        An = An + theta_a*theta_a';
        bn = bn+ r_t * theta_a;
        theta_hat = An\bn;
        % store regret
        regret(k, t) = model.best_arm_reward() - r_t;
        norm_dist(k, t) = norm(theta_hat - model.real_theta, 2);
    end
    %figure(2);
    %plot(1:T, choice_history);
    %title(alg_name)
end
delete(h);

% compute average (over sim) of the algorithm performance and plot it
regret = cumsum(regret,2);
mean_norms = sum(norm_dist)/nb_simu;
mean_regret = sum(regret)/nb_simu;

figure(1);
subplot(121);
plot(mean_norms, 'LineWidth', 2, 'DisplayName', alg_name);
hold on;
ylabel('$\|\hat{\theta} - \theta\|_{2}$','Interpreter','latex');
xlabel('Rounds');
legend show
subplot(122);
plot(cumsum(mean_regret), 'LineWidth', 2, 'DisplayName', alg_name);
hold on;
ylabel('Cumulative Regret');
xlabel('Rounds');
legend show
%%
alg_name = 'Random';

fprintf('Simulating %s...', alg_name);
regret = zeros(nb_simu, T);
norm_dist = zeros(nb_simu, T);
h = waitbar(0,'Initializing waitbar...');
for k = 1:nb_simu
    waitbar(k/nb_simu,h,sprintf('%d/%d', k, nb_simu));
    
    An = lambda * eye(d,d);
    bn = zeros(d,1);
    theta_hat = An\bn;
    for t = 1:T
        a_t = randi(n_a); % algorithm picks the action
        r_t = model.reward(a_t); % get the reward
        
        % do something (update algorithm)
        
        % store regret
        regret(k, t) = model.best_arm_reward() - r_t;
        norm_dist(k, t) = norm(theta_hat - model.real_theta, 2);
    end
end
delete(h);

% compute average (over sim) of the algorithm performance and plot it
regret = cumsum(regret,2);
mean_norms = sum(norm_dist)/nb_simu;
mean_regret = sum(regret)/nb_simu;

figure(1);
subplot(121);
plot(mean_norms, 'LineWidth', 2, 'DisplayName', alg_name);
ylabel('$\|\hat{\theta} - \theta\|_{2}$','Interpreter','latex');
xlabel('Rounds');

subplot(122);
plot(cumsum(mean_regret), 'LineWidth', 2, 'DisplayName', alg_name);
ylabel('Cumulative Regret');
xlabel('Rounds');

%%
epsilon = 0.01;
lambda = 1;
alpha_t = 1;
alg_name = ['LinUCB with epsilon=',num2str(epsilon)];

fprintf('Simulating %s...', alg_name);

regret = zeros(nb_simu, T);
norm_dist = zeros(nb_simu, T);
Z = model.features;


h = waitbar(0,'Initializing waitbar...');
for k = 1:nb_simu
    waitbar(k/nb_simu,h,sprintf('%d/%d', k, nb_simu));
    choice_history = zeros(T,1);
    
    An = lambda * eye(d,d);
    bn = zeros(d,1);
    theta_hat = An\bn;
    for t = 1:T
        if rand < epsilon
            a_t = randi(n_a);
        else
            beta = alpha_t * sum((Z/An).*Z,2);
            [~,a_t] = max(Z * theta_hat + beta); % algorithm picks the action
        end
        r_t = model.reward(a_t); % get the reward
        choice_history(t)=a_t;
        % do something (update algorithm)
        theta_a = Z(a_t,:)';
        An = An + theta_a*theta_a';
        bn = bn+ r_t * theta_a;
        theta_hat = An\bn;
        % store regret
        regret(k, t) = model.best_arm_reward() - r_t;
        norm_dist(k, t) = norm(theta_hat - model.real_theta, 2);
    end
    figure(3);
    plot(1:T, choice_history);
    title(alg_name)
end
delete(h);

% compute average (over sim) of the algorithm performance and plot it
regret = cumsum(regret,2);
mean_norms = sum(norm_dist)/nb_simu;
mean_regret = sum(regret)/nb_simu;

figure(1);
subplot(121);
plot(mean_norms, 'LineWidth', 2, 'DisplayName', alg_name);
hold on;
ylabel('$\|\hat{\theta} - \theta\|_{2}$','Interpreter','latex');
xlabel('Rounds');
legend show
subplot(122);
plot(cumsum(mean_regret), 'LineWidth', 2, 'DisplayName', alg_name);
hold on;
ylabel('Cumulative Regret');
xlabel('Rounds');
legend show


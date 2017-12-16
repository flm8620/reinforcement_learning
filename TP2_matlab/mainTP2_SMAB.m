%% Build your own bandit problem
clear all;
close all;
clc;

% this is an example, please change the parameters or arms!
Arm1 = armBernoulli(0.1);
Arm2 = armBernoulli(0.1);
Arm3 = armBernoulli(0.2);
Arm4 = armBernoulli(0.2);
Arm5 = armBernoulli(0.3);
Arm6 = armBernoulli(0.4);
Arm7 = armBernoulli(0.9);

% Arm1 = armBernoulli(0.1);
% Arm2 = armBernoulli(0.2);
% Arm3 = armBernoulli(0.3);
% Arm4 = armBernoulli(0.4);
% Arm5 = armBernoulli(0.5);
% Arm6 = armBernoulli(0.6);
% Arm7 = armBernoulli(0.7);

MAB = {Arm1, Arm2, Arm3, Arm4, Arm5, Arm6, Arm7};

% bandit : set of arms

NbArms = length(MAB);

Means = zeros(1, NbArms);
for i = 1:NbArms
    Means(i) = MAB{i}.mean;
end

% Display the means of your bandit (to find the best)
Means
[muMax,best_arm]=max(Means);

Cp = 0;
for i=1:7
    if i==best_arm
        continue 
    end
    x = MAB{i}.p;
    y = muMax;
    kl=x*log(x/y)+(1-x)*log((1-x)/(1-y));
    Cp = Cp + (y-x)/kl;
end

%% Comparison of the regret on one run of the bandit algorithm

T=10000; % horizon

[rew1, draws1] = UCB1(T,MAB);
reg1 = muMax*(1:T) - cumsum(rew1);
[rew2,draws2] = TS(T,MAB);
reg2 = muMax*(1:T) - cumsum(rew2);
[rew3,draws3] = random_strategy(T,MAB);
reg3 = muMax*(1:T) - cumsum(rew3);

% reg3 = naive strategy

% add oracle t -> C(p)log(t)
% kl = @(x,y) x*log(x/y)+(1-x)*log((1-x)/(1-y));

oracle = log(1:T)*Cp;
figure(1);
plot(1:T,reg1,1:T,reg2,1:T,oracle);
legend('UCB1','TS','oracle');


%% (Expected) regret curve for UCB and Thompson Sampling
reg1=zeros(1,T);
reg2=zeros(1,T);
reg3=zeros(1,T);
N=100;
h = waitbar(0,'Initializing waitbar...');
for i=1:N
    waitbar(i/N,h,sprintf('%d/%d', i, N));
    [rew1, draws1] = UCB1(T,MAB);
    reg1_i = muMax*(1:T) - cumsum(rew1);
    reg1 = reg1+reg1_i;
    [rew2,draws2] = TS(T,MAB);
    reg2_i = muMax*(1:T) - cumsum(rew2);
    reg2 = reg2+reg2_i;
end
delete(h);
reg1=reg1/N;
reg2=reg2/N;
reg3=reg3/N;
figure(2);
plot(1:T,reg1./log(1:T),1:T,reg2./log(1:T),1:T,Cp*ones(T,1));
legend('UCB1','TS','oracle');
title('regret/log(t)')
%% Non parametric bandits
clear all;
close all;
clc;

% this is an example, please change the parameters or arms!
Arm1 = armBeta(2,2);
Arm2 = armBeta(2,5);
Arm3 = armBeta(1,3);
Arm4 = armBeta(0.5,0.5);
Arm5 = armBeta(5,2);
Arm6 = armBeta(3,4);
Arm7 = armBeta(1,1);

MAB = {Arm1, Arm2, Arm3, Arm4, Arm5, Arm6, Arm7};

% bandit : set of arms

NbArms = length(MAB);

Means = zeros(1, NbArms);
for i = 1:NbArms
    Means(i) = MAB{i}.mean;
end

% Display the means of your bandit (to find the best)
Means
[muMax,best_arm]=max(Means);
Cp = 0;
for i=1:7
    if i==best_arm
        continue 
    end
    a = MAB{i}.a;
    b = MAB{i}.b;
    ap = MAB{best_arm}.a;
    bp = MAB{best_arm}.b;
    p_star = muMax;
    pa = MAB{i}.mean;
    kl=log(beta(ap,bp)/beta(a,b))-(ap-a)*psi(a)-(bp-b)*psi(b)+(ap-a+bp-b)*psi(a+b);
    Cp = Cp + (p_star-pa)/kl;
end
%% (Expected) regret curve for UCB and Thompson Sampling

T=10000; % horizon

reg1=zeros(1,T);
reg2=zeros(1,T);
reg3=zeros(1,T);
N=100;
h = waitbar(0,'Initializing waitbar...');
for i=1:N
    waitbar(i/N,h,sprintf('%d/%d', i, N));
    [rew1, draws1] = UCB1(T,MAB);
    reg1_i = muMax*(1:T) - cumsum(rew1);
    reg1 = reg1+reg1_i;
    [rew2,draws2] = TS_general(T,MAB);
    reg2_i = muMax*(1:T) - cumsum(rew2);
    reg2 = reg2+reg2_i;
end
delete(h);
reg1=reg1/N;
reg2=reg2/N;
reg3=reg3/N;




figure(3);
plot(1:T,reg1./log(1:T),1:T,reg2./log(1:T),1:T,Cp*ones(T,1));
legend('UCB1','TS','oracle');
title('regret/log(t)')
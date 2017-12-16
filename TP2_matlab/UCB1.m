function [rew,draws] = UCB1(T,MAB)
%UCB Summary of this function goes here
%   Detailed explanation goes here
rew=zeros(1,T);
draws=zeros(1,T);
N = size(MAB,2);
Sa = zeros(N,1);
Na = zeros(N,1);
rho = 0.2;
for i=1:N
    rew(i)=MAB{i}.sample();
    Sa(i)=rew(i);
    Na(i)=Na(i)+1;
    draws(i)=i;
end

for t=N+1:T
    score = zeros(N,1);
    for i=1:N
        score(i) = Sa(i)/Na(i)+rho*sqrt(log(t)/2/Na(i));
    end
    [~,ind]=max(score);
    rew(t)=MAB{ind}.sample();
    draws(t)=ind;
    Sa(ind)=Sa(ind)+rew(t);
    Na(ind)=Na(ind)+1;
end
end


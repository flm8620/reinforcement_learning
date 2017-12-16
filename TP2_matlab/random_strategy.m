function [rew,draws] = random_strategy(T,MAB)
rew=zeros(1,T);
draws=zeros(1,T);

N = size(MAB,2);
for t=1:T
    ind = randi(N);
    rew(t)=MAB{ind}.sample();
    draws(t)=ind;
end

end


function [rew, draws] = TS_general(T,MAB)
rew=zeros(1,T);
draws=zeros(1,T);
N = size(MAB,2);
Na = zeros(N,1);
Sa = zeros(N,1);

for t=1:T
    score = zeros(N,1);
    for i=1:N
        score(i) = betarnd(Sa(i)+1,Na(i)-Sa(i)+1);
    end
    [~,ind]=max(score);
    rew(t)=MAB{ind}.sample();
    draws(t)=ind;
    if rand < rew(t)
        r = 1;
    else
        r = 0;
    end
    Sa(ind)=Sa(ind)+r;
    Na(ind)=Na(ind)+1;
end
end


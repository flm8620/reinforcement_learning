p = zeros(3,2,3);
% p(x0,a0,x1)
p(1,1,1)=0.45;
p(1,1,3)=0.55;
p(1,2,3)=1.0;
p(2,1,3)=1.0;
p(2,2,1)=0.5;
p(2,2,2)=0.4;
p(3,1,1)=0.6;
p(3,1,3)=0.4;
p(3,2,2)=0.9;
p(3,2,3)=0.1;

R = zeros(3,2);
R(1,1)=-0.4;
R(1,2)=0;
R(2,1)=2;
R(2,2)=0;
R(3,1)=-1;
R(3,2)=-0.5;

gamma=0.95;
epsilon = 0.01;

%% optim policy evaluation to get V_optim
d_optim = [2,1,2];
P = zeros(3,3);
for x=1:3
    for y=1:3
        P(x,y) = p(x,d_optim(x),y);
    end
end
r_pi = [R(1,d_optim(1)),R(2,d_optim(2)),R(3,d_optim(3))];
V_optim = (eye(3) - gamma*P)\r_pi';

%% Q2 value iteration
max_iter = 1000;
V = zeros(3, 1);
gamma = 0.95;
count = 0;
history_diff = zeros(max_iter,1);
history_error = zeros(max_iter,1);
while true
    count=count+1;
    V_next = zeros(3, 1);
    for x = 1:3
        value1 = R(x,1) + gamma * dot(reshape(p(x,1,:),[],1),V);
        value2 = R(x,2) + gamma * dot(reshape(p(x,2,:),[],1),V);
        V_next(x) = max(value1,value2);
    end
    history_diff(count) = max(abs(V_next-V));
    history_error(count) = max(abs(V_next-V_optim));
    if history_diff(count) < epsilon
        break;
    end
    V=V_next;
end
figure(1);
plot(1:count,log(history_diff(1:count)));
title('log(||v^{k+1}-v^k||_{\infty})');
print('-depsc','v_diff');
figure(2);
plot(1:count,log(history_error(1:count)));
title('log(||v^{*}-v^k||_{\infty})');
print('-depsc','v_error');

%% Q3
max_iter = 1000;
history_diff = zeros(max_iter,1); % total history difference during entire evaulation
history_error = zeros(max_iter,1);

V = zeros(3, 1);

d = [randi([1,2]),randi([1,2]),randi([1,2])];
old_d = d;
count = 1;
while true % for each policy d
    while true
        %fprintf('%d\n',count);
        V_next = zeros(3, 1);
        for x = 1:3
            a = d(x);
            V_next(x) = R(x,a) + gamma * dot(reshape(p(x,a,:),[],1),V);
        end
        difference = max(abs(V_next-V));
        if difference < epsilon
            break;
        end
        V=V_next;
        count=count+1;
        
        % record diff and error
        history_diff(count) = difference;
        history_error(count) = max(abs(V_next-V_optim));
    end
    
    for x=1:3
        Q = zeros(2,1);
        for i=1:2
            proba_destination = reshape(p(x,i,:),[],1);
            Q(i) = R(x,i) + gamma * (proba_destination'*V);
        end
        [~,ind] = max(Q);
        d(x) = ind;
    end
    
    if all(d == old_d)
        break;
    end
    old_d = d;
end

figure(3);
plot(1:count,log(history_diff(1:count)));
title('PI: log(||v^{k+1}-v^k||_{\infty})');
print('-depsc','v_diff_policy');
figure(4);
plot(1:count,log(history_error(1:count)));
title('PI: log(||v^{*}-v^k||_{\infty})');
print('-depsc','v_error_policy');






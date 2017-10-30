function render_policy(env, pol)
dim = 200;
figure(); hold on;
set(gca,'Ydir','reverse')
for s=1:env.n_states
    r = env.state2coord{s}(1);
    c = env.state2coord{s}(2);
    draw_square_pol(10 + c * (dim + 4), 10 + r * (dim + 4), dim, pol{s}, env.state_actions{s});
end
axis off;
hold off;
end

function draw_square_pol(x,y,dim,pol,actions)

if length(pol) == 1
    d = -ones(length(actions));
    d(actions == pol) = 1;
else
    d = pol;
end

v1 = [x, y; x + dim, y; x + dim, y+dim; x, y+dim];
f1 = [1,2,3,4];
patch('Faces',f1,'Vertices',v1,'FaceColor','white', 'EdgeColor', 'k')

DIG = 2;
for j = 1:length(d)
    if j <= length(actions)
        action = actions(j);
        if action == 1
            p1 = [x + dim / 2., y + dim / 2.];
            p2 = [x + 3*dim / 4., y + dim / 2.];
        elseif action ==2
            p1 = [x + dim / 2., y + dim / 2.];
            p2 = [x + dim / 2., y + 3* dim / 4.];
        elseif action ==3
            p1 = [x + dim / 2., y + dim / 2.];
            p2 = [x+dim/4., y + dim/2.];
        elseif action ==4
            p1 = [x + dim / 2., y + dim / 2.];
            p2 = [x + dim / 2., y + dim / 4.];
        end
        
        if d(j) > 0
            dp = p2-p1;
            quiver(p1(1),p1(2),dp(1),dp(2),0, 'LineWidth', 2., 'MaxHeadSize', 1.5, 'Color', 'red')
            if ~isclose(d(j), 1)
                text(p2(1), p2(2), num2str(d(j),DIG));
            end
        end
    end
end

end

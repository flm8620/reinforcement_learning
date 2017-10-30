function render_q(env, q)
dim = 200;
figure();
for s=1:env.n_states
    r = env.state2coord{s}(1);
    c = env.state2coord{s}(2);
    draw_square_q(10 + c * (dim + 4), 10 + r * (dim + 4), dim, q(s,:), env.state_actions{s});
end
axis off;
end

function draw_square_q(x,y,dim,q,actions)
v1 = [x, y; x + dim, y; x + dim, y+dim; x, y+dim];
f1 = [1,2,3,4];
patch('Faces',f1,'Vertices',v1,'FaceColor','white', 'EdgeColor', 'k')

DIG = 2;
f1 = [1,2,3];
idx = 1;
for action = actions
    if action == 1
        v1 = [x + dim, y; x + dim / 2., y + dim / 2.; x + dim, y + dim];
        patch('Faces',f1,'Vertices',v1,'FaceColor','white', 'EdgeColor', 'k')
        text(x + 3 * dim / 4., y + dim / 2., num2str(q(idx),DIG));
    elseif action ==2
        v1 = [x, y + dim; x + dim / 2., y + dim / 2.; x + dim, y + dim];
        patch('Faces',f1,'Vertices',v1,'FaceColor','white', 'EdgeColor', 'k')
        text(x + dim / 2., y + 3 * dim / 4., num2str(q(idx),DIG));
    elseif action ==3
        v1 = [x, y; x + dim / 2., y + dim / 2.; x, y + dim];
        patch('Faces',f1,'Vertices',v1,'FaceColor','white', 'EdgeColor', 'k')
        text(x + dim / 4., y + dim / 2., num2str(q(idx),DIG));
    elseif action ==4
        v1 = [x + dim, y; x + dim / 2., y + dim / 2.; x, y];
        patch('Faces',f1,'Vertices',v1,'FaceColor','white', 'EdgeColor', 'k')
        text(x + dim / 2., y + dim / 4., num2str(q(idx),DIG));
    end
    idx = idx + 1;
end
set(gca,'Ydir','reverse')

end
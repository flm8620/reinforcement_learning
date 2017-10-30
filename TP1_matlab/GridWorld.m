classdef GridWorld
    properties
        grid
        state2coord
        coord2state
        action_names
        
        n_rows
        n_cols
        proba_succ
        
        n_states
        state_actions
        gamma
        
        render
    end
    methods
        function obj = GridWorld(grid, gamma, render)
            obj.grid = grid;
            obj.action_names = {'right', 'down', 'left', 'up'};
            
            obj.n_rows = size(obj.grid,1);
            obj.n_cols = max(cellfun(@(c) length(c), obj.grid));
            obj.coord2state = zeros(obj.n_rows, obj.n_cols);
            obj.state2coord = {};
            obj.n_states = 0;
            for i = 1:obj.n_rows
                for j = 1:obj.n_cols
                    if isempty(grid{i}{j}) || grid{i}{j} ~= 'x'
                        obj.n_states = obj.n_states + 1;
                        obj.coord2state(i,j) = obj.n_states;
                        obj.state2coord{obj.n_states} = [i,j];
                    else
                        obj.coord2state(i,j) = -1;
                    end
                end
            end
            
            obj.gamma = gamma;
            obj.state_actions = {};
            
            obj.proba_succ = 0.9;
            
            obj = obj.compute_available_actions();
            
            if nargin < 3
                obj.render = 0;
            else
                obj.render = render;
            end
        end
        
        function state = reset(obj)
            % Returns:
            %     An initial state randomly drawn from
            %     the initial distribution
        
            state = randi([1,obj.n_states], 1);
        end
        
        function [next_state, reward, term] = step(obj, state, action)
            % Args:
            %     state (int): the amount of good
            %     action (int): the action to be executed
            %
            % Returns:
            %     next_state (int): the state reached by performing the action
            %     reward (float): a scalar value representing the immediate reward
            %     absorb (boolean): True if the next_state is absorsing, False otherwise
        
            assert(any(obj.state_actions{state} == action))
            r = obj.state2coord{state}(1);
            c = obj.state2coord{state}(2);
            if ~isempty(obj.grid{r}{c}) && isnumeric(obj.grid{r}{c})
                % absorbing state
                next_state = state;
                reward = 0;
                term = 1;
            else
                failed = rand(1) > obj.proba_succ;
                if action == 1 % right
                    if failed
                        c = max(1, c - 1);
                    else
                        c = min(obj.n_cols,c+1);
                    end
                elseif action == 2 % down
                    if failed
                        r = max(1, r - 1);
                    else
                        r = min(obj.n_rows,r+1);
                    end
                elseif action == 3 % left
                    if failed
                        c = min(obj.n_cols, c + 1);
                    else
                        c = max(0,c-1);
                    end
                elseif action == 4 % up
                    if failed
                        r = min(obj.n_rows, r + 1);
                    else
                        r = max(0, r-1);
                    end
                end
                if ~isempty(obj.grid{r}{c}) && obj.grid{r}{c} == 'x'
                    % if the next state is a wall, stay in the same place
                    next_state = state;
                    r = obj.state2coord{state}(1);
                    c = obj.state2coord{state}(2);
                else
                    next_state = obj.coord2state(r,c);
                end
                
                if ~isempty(obj.grid{r}{c}) && isnumeric(obj.grid{r}{c})
                    reward = obj.grid{r}{c};
                    term = 1.;
                else
                    reward = 0.;
                    term = 0;
                end
            end
            
            if obj.render == 1
                obj.show(state, action, next_state, reward)
            end
        end
        
        function obj = compute_available_actions(obj)
            obj.state_actions = {};
            for i = 1:obj.n_rows
                for j = 1:obj.n_cols
                    if ~isempty(obj.grid{i}{j}) && isnumeric(obj.grid{i}{j})
                        % the only available action is right (arbitrary
                        % choice)
                        obj.state_actions{obj.coord2state(i,j)} = [1];
                    elseif isempty(obj.grid{i}{j})
                        actions = 1:4;
                        if i == 1
                            actions(actions == 4) = [];
                        end
                        if j == obj.n_cols
                            actions(actions == 1) = [];
                        end
                        if i == obj.n_rows
                            actions(actions == 2) = [];
                        end
                        if j == 1
                            actions(actions == 3) = [];
                        end
                        
                        actions2 = actions;
                        for a = actions2
                            r =i;
                            c =j;
                            if a == 1
                                c = min(obj.n_cols,c+1);
                            elseif a == 2
                                r = min(obj.n_rows,r+1);
                            elseif a == 3
                                c = max(0,c-1);
                            elseif a == 4
                                r = max(0, r-1);
                            end
                            if ~isempty(obj.grid{r}{c}) && obj.grid{r}{c} == 'x'
                                actions(actions == a) = [];
                            end
                        end
                        obj.state_actions{obj.coord2state(i,j)} = actions;
                    end
                end
            end
            
        end
        
        function show(obj, state, action, next_state, reward)
            dim = 200;
            clf;
            axis off;
            for s = 1:obj.n_states
                r = obj.state2coord{s}(1);
                c = obj.state2coord{s}(2);
                x = 10 + c * (dim + 4);
                y = 10 + r * (dim + 4);
                v1 = [x, y; x + dim, y; x + dim, y+dim; x, y+dim];
                f1 = [1,2,3,4];
                if ~isempty(obj.grid{r}{c}) && obj.grid{r}{c} ~= 'x'
                    patch('Faces',f1,'Vertices',v1,'FaceColor','blue', 'EdgeColor', 'k')
                    text(x+dim/2., y+dim/2., num2str(obj.grid{r}{c}, 1), 'FontSize', 14, 'Color', 'white')
                else
                    patch('Faces',f1,'Vertices',v1,'FaceColor','white', 'EdgeColor', 'k')
                end
            end
            set(gca,'Ydir','reverse')
            
            r00 = obj.state2coord{state}(1);
            c00 = obj.state2coord{state}(2);
            c0 = 10 + r00 * (dim + 4);
            r0 = 10 + c00 * (dim + 4);
            x0 = r0 + dim / 2.;
            y0 = c0 + dim / 2.;
            r11 = obj.state2coord{next_state}(1);
            c11 = obj.state2coord{next_state}(2);
            c1 = 10 + r11 * (dim + 4);
            r1 = 10 + c11 * (dim + 4);
            x1 = r1 + dim / 2.;
            y1 = c1 + dim / 2.;
            
            
            t = linspace(0, 2*pi);
            r = dim/4.;
            a = r*cos(t) + x1;
            b = r*sin(t) + y1;
            patch(a, b, 'g')
            
            text(2 * dim, dim/2. + (length(obj.grid)+1) * (dim + 4), ['r: ', num2str(reward)], 'FontSize', 14)
            text(4 * dim, dim/2. + (length(obj.grid)+1) * (dim + 4), ['action: ', obj.action_names{action}], 'FontSize', 14)
        end
        
    end
end

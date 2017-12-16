classdef LinearMABModel < handle
    
    properties
        local_random
        features
        real_theta
        noise
    end
    
    methods
        function obj = LinearMABModel(random_state, noise)
            obj.local_random=RandStream('mt19937ar', 'Seed', random_state);
            obj.noise = noise;
        end
        
        function reward = reward(obj, action)
            assert(1<= action && action <= obj.n_actions());
            reward = obj.features(action,:) * obj.real_theta + obj.noise * obj.local_random.randn(1);
        end
        
        function max_reward =  best_arm_reward(obj)
            D = obj.features * obj.real_theta;
            max_reward = max(D);
        end
        
        function d = n_features(obj)
            d = size(obj.features, 2);
        end
        
        function n = n_actions(obj)
            n = size(obj.features, 1);
        end
        
    end % end methods
    
end % end class

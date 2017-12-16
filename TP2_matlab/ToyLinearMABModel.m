classdef ToyLinearMABModel < LinearMABModel
    
    methods
        function obj = ToyLinearMABModel(n_features, n_actions, random_state, noise)
            obj = obj@LinearMABModel(random_state, noise);
            obj.features = obj.local_random.rand(n_actions, n_features) - 0.5;
            obj.real_theta = obj.local_random.rand(n_features, 1) - 0.5;
        end
    end
    
end

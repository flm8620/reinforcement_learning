classdef ColdStartMovieLensModel < LinearMABModel
    
    methods
        function obj = ColdStartMovieLensModel(random_state, noise)
            obj = obj@LinearMABModel(random_state, noise);
            obj.features = csvread('movielens/Vt.csv')';
            obj.real_theta = obj.local_random.rand(obj.n_features(), 1);
        end
    end % end methods
    
end %end class

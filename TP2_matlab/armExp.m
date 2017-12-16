classdef armExp<handle
    % arm with trucated exponential distribution
    % https://en.wikipedia.org/wiki/Truncated_distribution
    % https://en.wikipedia.org/wiki/Exponential_distribution
    % http://lagrange.math.siu.edu/Olive/ch4.pdf
    
    properties
        lambda % parameter of the exponential distribution
        B % upper bound of the distribution (lower is 0)
        mean % expectation of the arm
        var % variance of the arm 
    end
    
    methods
        function self = armExp(lambda, B)
            self.B = B;
            self.lambda=lambda;
            self.mean = (1/lambda)*(1-exp(-lambda));
            self.var = 1; % compute it yourself !
        end
        
        function [reward] = sample(self)
            % Inverse transform sampling
            % https://en.wikipedia.org/wiki/Inverse_transform_sampling
            u = rand();
            reward = - log(1. - (1. - exp(- self.lambda * self.B)) * u) / self.lambda;
        end
                
    end    
end
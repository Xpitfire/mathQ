using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace mathQ.CSharp.NeuralNet.Function
{
    public sealed class NeuronCalculation
    {
        public static double PerceptronFunction(IReadOnlyList<double> values, IReadOnlyList<double> weights, IReadOnlyList<double> biases)
        {
            if (values.Count() != weights.Count() 
                && values.Count() != biases.Count())
            {
                throw new InvalidOperationException(
                    "Cannot evaluate Perceptron! Weights, biases and input values must have the same length.");
            }
            return SigmoidFunction(values.Select((t, i) => WeightFunction(t, weights[i], biases[i])).Sum());
        }

        public static double SigmoidFunction(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public static double WeightFunction(double value, double weight, double bias)
        {
            return value * weight + bias;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public struct Perceptron : IPerceptron
    {
        public IEnumerable<double> InputValues { get; set; }
        public double OutputValue { get; set; }
        public IEnumerable<double> Weights { get; set; }
        public IEnumerable<double> Biases { get; set; }
        public Func<IEnumerable<double>, double> ValueTransformation { get; set; }
        
        public static double PerceptronFunction(double value, double weight, double bias)
        {
            return value*weight + bias;
        }
        
        public static double SigmoidFunction(double value)
        {
            return 1.0/(1.0 + Math.Exp(-value));
        }

        public static double OutputFunction(double value, double weight, double bias)
        {
            return SigmoidFunction(PerceptronFunction(value, weight, bias));
        }
        
        private double Evaluate(IReadOnlyList<double> values, IReadOnlyList<double> weights, IReadOnlyList<double> biases)
        {
            if (values.Count() != weights.Count() && values.Count() != biases.Count())
            {
                throw new InvalidOperationException(
                    "Cannot evaluate Perceptron! Weights, biases and input values must have the same length.");
            }
            return values.Select((t, i) => OutputFunction(t, weights[i], biases[i])).Sum();
        }

    }
}

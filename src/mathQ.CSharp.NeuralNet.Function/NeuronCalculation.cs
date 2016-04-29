using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace mathQ.CSharp.NeuralNet.Function
{
    public sealed class NeuronCalculation
    {
        public static double ActivationFunction(IList<double> values, IList<double> weights, double bias)
        {
            if (values.Count != weights.Count)
            {
                throw new InvalidOperationException(
                    "Cannot evaluate Perceptron! Weights and input values must have the same length.");
            }
            return SigmoidFunction(values.Select((t, i) => t*weights[i]).Sum() + bias);
        }

        public static double ZVectorFunction(IList<double> values, IList<double> weights, double bias)
        {
            return values.Select((t, i) => t*weights[i]).Sum() + bias;
        }

        public static double SigmoidFunction(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public static double SigmoidPrimeFunction(double value)
        {
            return NeuronCalculation.SigmoidFunction(value) *
                   (1 - NeuronCalculation.SigmoidFunction(value));
        }
    }
}

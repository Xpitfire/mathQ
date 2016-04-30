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
            return SigmoidFunction(ZVectorFunction(values, weights, bias));
        }

        public static double ZVectorFunction(IList<double> values, IList<double> weights, double bias)
        {
            return values.Select((t, i) => t*weights[i]).Sum() + bias;
        }

        public static double SigmoidFunction(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }
        
        public static double GradientFunction(double value)
        {
            return value*(1 - value);
        }

        public static IList<double> VectorSubtraction(IList<double> v1, IList<double> v2)
        {
            return v1.Select((t, i) => t - v2[i]).ToList();
        }

        public static IList<double> GradientSigmoidFunction(IList<double> values)
        {
            return values.Select(v => GradientFunction(SigmoidFunction(v))).ToList();
        }

        public static double GradientSigmoidFunction(double value)
        {
            return GradientFunction(SigmoidFunction(value));
        }

        public static double VectorDotProduct(IList<double> v1, IList<double> v2)
        {
            return v1.Select((t, i) => t*v2[i]).Sum();
        }

        public static double VectorDotProduct(double scalar, IList<double> v2)
        {
            var v1 = new List<double>(v2.Count);
            for (var i = 0; i < v2.Count; i++)
            {
                v1.Add(scalar);
            }
            return v1.Select((t, i) => t * v2[i]).Sum();
        }

        public static IList<double> VectorScalarMultiplication(double prevLayerDelta, IList<double> outputValues)
        {
            return outputValues.Select(d => d*prevLayerDelta).ToList();
        }

        public static IList<double> VectorAdd(IList<double> v1, IList<double> v2)
        {
            return v1.Select((t, i) => t + v2[i]).ToList();
        }

        public static double CostFunction(double val1, double val2)
        {
            return val1 - val2;

        }

        public static IList<double> CostFunction(IList<double> v1, IList<double> v2)
        {
            return VectorSubtraction(v1, v2);

        }

        public static IList<double> VectorMultiplication(IList<double> v1, IList<double> v2)
        {
            return v1.Select((v, i) => v*v2[i]).ToList();
        }

        public static double SquaredErrorFunction(double idealValue, double actualValue)
        {
            return .5*Math.Pow(idealValue - actualValue, 2);
        }
    }
}

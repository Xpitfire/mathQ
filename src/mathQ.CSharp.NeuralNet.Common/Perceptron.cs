using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class Perceptron : IPerceptron
    {
        public IList<double> InputValues { get; set; }
        public double OutputValue { get; set; }
        public NeuronFunction<double, double> TransformationFunction { get; set; }
        public IList<double> Weights { get; set; }
        public double Bias { get; set; } = 1.0;
        public PerceptronFunction PerceptronFunction { private get; set; }

        public Perceptron()
        {
            TransformationFunction = values => PerceptronFunction(values, Weights, Bias);
        }
        
        public void Evaluate()
        {
            OutputValue = TransformationFunction(InputValues);
        }
    }
}

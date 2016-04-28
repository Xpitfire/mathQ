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
        public IEnumerable<double> InputValues { get; set; }
        public double OutputValue { get; set; }
        public NeuronFunction<double, double> TransformationFunction { get; set; }
        public IEnumerable<double> Weights { get; set; }
        public IEnumerable<double> Biases { get; set; }
        public PerceptronFunction PerceptronFunction { private get; set; }

        public Perceptron()
        {
            TransformationFunction = values => PerceptronFunction(values, Weights.ToList(), Biases.ToList());
        }
        
        public void Evaluate()
        {
            OutputValue = TransformationFunction(InputValues.ToList());
        }
    }
}

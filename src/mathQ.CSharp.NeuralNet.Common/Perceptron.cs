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
        public PerceptronFunction ValueTransformation { private get; set; }
        public void Evaluate()
        {
            OutputValue = ValueTransformation(
                InputValues.ToList(), 
                Weights.ToList(), 
                Biases.ToList());
        }

    }
}

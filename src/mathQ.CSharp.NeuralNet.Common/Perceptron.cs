using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using mathQ.CSharp.NeuralNet.Function;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class Perceptron : IPerceptron
    {
        public IList<double> InputValues { get; set; }
        public double OutputValue { get; private set; }
        public IList<double> Weights { get; set; }
        public double Bias { get; set; } = 1.0;
        
        public void Evaluate()
        {
            OutputValue = 1.0 / (1.0 + Math.Exp(
                -InputValues.Select((t, i) => t * Weights[i]).Sum() + Bias));
        }
    }
}

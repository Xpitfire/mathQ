using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public delegate double PerceptronFunction(IReadOnlyList<double> values, IReadOnlyList<double> weights, IReadOnlyList<double> biases);

    public interface IPerceptron : INeuron<double, double>
    {
        IEnumerable<double> Weights { get; set; }
        IEnumerable<double> Biases { get; set; }
        PerceptronFunction ValueTransformation { set; }
        void Evaluate();
    }
}

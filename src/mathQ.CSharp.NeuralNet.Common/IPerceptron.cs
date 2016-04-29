using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public delegate double PerceptronFunction(IList<double> values, IList<double> weights, double bias);

    public interface IPerceptron : INeuron<double, double>
    {
        IList<double> Weights { get; set; }
        double Bias { get; set; }
        PerceptronFunction PerceptronFunction { set; }
    }
}

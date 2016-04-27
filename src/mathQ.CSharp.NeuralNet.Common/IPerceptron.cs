using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface IPerceptron : INeuron<double, double>
    {
        IEnumerable<double> Weights { get; set; }
        IEnumerable<double> Biases { get; set; }
        Func<IEnumerable<double>, double> ValueTransformation { set; }
    }
}

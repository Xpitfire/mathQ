using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralHiddenLayer : INeuralLayer<double, double>
    {
        IEnumerable<IPerceptron> Perceptrons { get; set; }

        void Compute();
        void Evaluate();
    }
}

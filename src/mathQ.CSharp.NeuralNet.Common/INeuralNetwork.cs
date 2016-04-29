using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralNetwork<TInput, TOutput>
    {
        INeuralInputLayer<TInput> InputLayer { get; set; }
        IList<INeuralHiddenLayer> HiddenLayers { get; set; }
        INeuralOutputLayer<TOutput> OutputLayer { get; set; }

        TOutput Evaluate(TInput inputValue);
    }
}

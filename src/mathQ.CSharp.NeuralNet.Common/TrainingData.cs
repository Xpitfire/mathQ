using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class TrainingData<TInput, TOutput>
    {
        public INeuralInputLayer<TInput> InputLayer { get; set; }
        public IList<INeuralHiddenLayer> HiddenLayers { get; set; }
        public INeuralOutputLayer<TOutput> OutputLayer { get; set; }
    }
}

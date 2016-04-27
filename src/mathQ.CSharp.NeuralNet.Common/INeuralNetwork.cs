using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralNetwork<TInput, TOutput>
    {
        INeuralInputLayer<TInput> InputLayer { get; }
        IList<INeuralHiddenLayer> HiddenLayers { get; }
        INeuralOutputLayer<TOutput> OutputLayer { get; }

        TOutput Evaluate(IEnumerable<TInput> inputValue);
        void Train(TrainingData<TInput, TOutput> trainingDataset);
    }
}

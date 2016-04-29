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

        TOutput Evaluate(TInput inputValue);
        void Train(ITrainingData<TInput, TOutput> trainingDataset);
    }
}

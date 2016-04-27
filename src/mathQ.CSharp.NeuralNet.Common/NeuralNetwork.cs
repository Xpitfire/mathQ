using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralNetwork<TInput, TOutput> : INeuralNetwork<TInput, TOutput>
    {
        public INeuralInputLayer<TInput> InputLayer { get; private set; }
        public IList<INeuralHiddenLayer> HiddenLayers { get; private set; }
        public INeuralOutputLayer<TOutput> OutputLayer { get; private set; }

        public TOutput Evaluate(IEnumerable<TInput> values)
        {
            return default(TOutput); // TODO
        }
        
        public void Train(TrainingData<TInput, TOutput> trainingDataset)
        {
            InputLayer = trainingDataset.InputLayer;
            HiddenLayers = trainingDataset.HiddenLayers;
            OutputLayer = trainingDataset.OutputLayer;
        }
    }
}

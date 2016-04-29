using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface ITrainingData<TInput, TOutput>
    {
        INeuralInputLayer<TInput> InputLayer { get; set; }
        IList<INeuralHiddenLayer> HiddenLayers { get; set; }
        INeuralOutputLayer<TOutput> OutputLayer { get; set; }
        int MaxEpochs { get; set; }
        double LearningRate { get; set; }
        double Momentum { get; set; }

        void Randomize(double minWeight = -10.0, double maxWeight = 10.0, double minBias = 1, double maxBias = 10);
        void Train(IList<TInput> trainingValues);
        void InitializePresets(double[][][] values);
    }
}

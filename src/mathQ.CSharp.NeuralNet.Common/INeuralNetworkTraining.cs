using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralNetworkTraining<TInput, TOutput>
    {
        INeuralNetwork<TInput, TOutput> NeuralNetwork { get; set; }

        int MaxEpochs { get; set; }
        double LearningRate { get; set; }
        double Momentum { get; set; }

        void Initialize(int initialNumberOfInputValues, params int[] numberOfPerceptronsPerHiddenLayer);
        void Randomize();
        void Train(IList<Tuple<TInput, double>> trainingData);
        void InitializePresets(double[][][] values);
    }
}

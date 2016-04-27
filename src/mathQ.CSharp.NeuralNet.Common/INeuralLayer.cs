using System.Collections.Generic;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralLayer<TInput, TOutput>
    {
        IEnumerable<TInput> InputValues { get; set; }
        IEnumerable<TOutput> OutputValues { get; set; }

        event NeuronSignal NeuronSignal;
    }

    
}
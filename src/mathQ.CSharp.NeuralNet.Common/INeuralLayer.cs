using System.Collections.Generic;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralLayer<TInput, TOutput>
    {
        IList<TInput> InputValues { get; set; }
        IList<TOutput> OutputValues { get; set; }
    }

    
}
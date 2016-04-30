using System.Collections.Generic;

namespace mathQ.CSharp.NeuralNet.Common
{
    public delegate TOutput NeuronFunction<TInput, out TOutput>(IList<TInput> values);

    public interface INeuron<TInput, out TOutput>
    {
        IList<TInput> InputValues { get; set; }
        TOutput OutputValue { get; }

        void Evaluate();
    }
}

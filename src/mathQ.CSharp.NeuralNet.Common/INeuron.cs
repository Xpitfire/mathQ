using System.Collections.Generic;

namespace mathQ.CSharp.NeuralNet.Common
{
    public delegate TOutput NeuronFunction<TInput, out TOutput>(IList<TInput> values);

    public interface INeuron<TInput, TOutput>
    {
        IList<TInput> InputValues { get; set; }
        TOutput OutputValue { get; set; }
        NeuronFunction<TInput, TOutput> TransformationFunction { set; }

        void Evaluate();
    }
}

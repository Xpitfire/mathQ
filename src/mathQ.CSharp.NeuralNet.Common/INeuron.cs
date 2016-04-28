using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace mathQ.CSharp.NeuralNet
{
    public delegate TOutput NeuronFunction<in TInput, out TOutput>(IReadOnlyList<TInput> values);

    public interface INeuron<TInput, TOutput>
    {
        IEnumerable<TInput> InputValues { get; set; }
        TOutput OutputValue { get; set; }
        NeuronFunction<TInput, TOutput> TransformationFunction { set; }

        void Evaluate();
    }
}

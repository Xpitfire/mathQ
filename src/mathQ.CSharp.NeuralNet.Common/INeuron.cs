using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace mathQ.CSharp.NeuralNet
{
    public interface INeuron<TInput, TOutput>
    {
        IEnumerable<TInput> InputValues { get; set; }
        TOutput OutputValue { get; set; }
    }
}

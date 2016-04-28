using System;

namespace mathQ.CSharp.NeuralNet.Common
{
    public delegate void NeuronSignal<TType>(object sender, NeuronSignalArgs<TType> args);

    public class NeuronSignalArgs<TType> : EventArgs
    {
        public TType Value { get; set; }
    }

}
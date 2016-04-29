using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralNetwork<TInput, TOutput> : INeuralNetwork<TInput, TOutput>
    {
        public INeuralInputLayer<TInput> InputLayer { get; set; }
        public IList<INeuralHiddenLayer> HiddenLayers { get; set; }
        public INeuralOutputLayer<TOutput> OutputLayer { get; set; }

        public TOutput Evaluate(IList<TInput> value)
        {
            InputLayer.InputValues = value;
            InputLayer.Transform();
            var curInputValues = InputLayer.OutputValues;
            var nextLayer = HiddenLayers?.GetEnumerator();
            while (nextLayer?.MoveNext() ?? false)
            {
                var curLayer = nextLayer.Current;
                curLayer.InputValues = curInputValues;
                curLayer.Evaluate();
                curInputValues = curLayer.OutputValues;
            }
            OutputLayer.InputValues = curInputValues;
            OutputLayer.Evaluate();
            OutputLayer.Transform();
            return OutputLayer.OutputValue;
        }
        
    }
}

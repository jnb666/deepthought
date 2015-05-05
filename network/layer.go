package network

import (
	//"fmt"
	"github.com/jnb666/deepthought/blas"
	"math"
)

// Layer interface type represents one layer in the network.
type Layer interface {
	FeedForward(in blas.Matrix) blas.Matrix
	BackProp(err blas.Matrix, eta float64) blas.Matrix
	Weights() blas.Matrix
	Gradient() blas.Matrix
	Cost(t blas.Matrix) float64
}

type layer struct {
	activ    Activation
	bias     blas.Matrix // column vector of 1s
	input    blas.Matrix // Z matrix of value at each node [samples, nin+1]
	output   blas.Matrix // return value at each node [samples, nout]
	weights  blas.Matrix // W outgoing weight matrix  [nout, nin+1]
	gradient blas.Matrix // gradient of weight matrix [nout, nin+1]
	deriv    blas.Matrix // Fp matrix of derivative of activation fn [samples, nin]
	delta    blas.Matrix // D matrix of errors at each node [samples, nout]
}

// AddLayer method adds a new input or hidden layer to the network.
func (n *Network) AddLayer(nin, nout int, a Activation) {
	batch := n.batchSize
	l := &layer{
		activ:    a,
		bias:     n.bias,
		input:    blas.New(batch, nin+1),
		output:   blas.New(batch, nout),
		weights:  blas.New(nout, nin+1),
		gradient: blas.New(nout, nin+1),
	}
	if a.Func != nil {
		l.deriv = blas.New(batch, nin)
	}
	if n.layers > 0 {
		l.delta = blas.New(batch, nin)
	}
	n.Add(l)
}

func (l *layer) Weights() blas.Matrix { return l.weights }

func (l *layer) Gradient() blas.Matrix { return l.gradient }

func (l *layer) Cost(t blas.Matrix) float64 { panic("no cost for input or hidden layer!") }

func (l *layer) FeedForward(in blas.Matrix) blas.Matrix {
	l.bias.Reshape(in.Rows(), 1)
	if l.activ.Func != nil {
		l.activ.Func.Apply(in, l.input)
		l.activ.Deriv.Apply(in, l.deriv)
		l.input.Join(l.input, l.bias)
	} else {
		l.input.Join(in, l.bias)
	}
	return l.output.Mul(l.input, l.weights, false, true, false)
}

func (l *layer) BackProp(err blas.Matrix, eta float64) blas.Matrix {
	// update weights
	l.gradient.Mul(err, l.input, true, false, false)  // [samples, nout].T x [samples, nin+1]
	l.gradient.Scale(-eta / float64(l.output.Rows())) // [nout, nin+1]
	l.weights.Add(l.weights, l.gradient)              // [nout, nin+1]
	// propagate error backward
	if l.delta != nil {
		wnb := l.weights.Slice(0, l.weights.Cols()-1) // [nout, nin]
		l.delta.Mul(wnb, err, true, true, true)       // [nout, nin].T x [samples, nout].T
		l.delta.MulElem(l.delta, l.deriv)             // [samples, nin]
	}
	return l.delta
}

type outLayer struct {
	activ  Activation
	values blas.Matrix // Z matrix of values at each node [samples, nodes]
	delta  blas.Matrix // D matrix of errors at each node [nodes, samples]
	deriv  blas.Matrix // Fp matrix of derivative of activation fn [nin, samples]
	temp   blas.Matrix
}

func newOutLayer(batch, nodes int, a Activation) *outLayer {
	return &outLayer{
		activ:  a,
		values: blas.New(batch, nodes),
		delta:  blas.New(batch, nodes),
		deriv:  blas.New(batch, nodes),
		temp:   blas.New(batch, nodes),
	}
}

func (l *outLayer) Weights() blas.Matrix { panic("no weights for output layer!") }

func (l *outLayer) Gradient() blas.Matrix { panic("no gradient for output layer!") }

func (l *outLayer) FeedForward(in blas.Matrix) blas.Matrix {
	if l.activ.Func != nil {
		l.activ.Func.Apply(in, l.values)
		l.activ.Deriv.Apply(in, l.deriv)
	} else {
		l.values = in
	}
	return l.values
}

type quadraticOutput struct{ *outLayer }

// QuadraticOutput method appends a quadratic cost output layer to the network.
func (n *Network) QuadraticOutput(nodes int, a Activation) {
	n.Add(&quadraticOutput{newOutLayer(n.batchSize, nodes, a)})
}

func (l *quadraticOutput) BackProp(target blas.Matrix, eta float64) blas.Matrix {
	l.delta.Sub(l.values, target)
	return l.delta.MulElem(l.delta, l.deriv)
}

func (l *quadraticOutput) Cost(target blas.Matrix) float64 {
	l.temp.Sub(l.values, target)
	l.temp.MulElem(l.temp, l.temp)
	return 0.5 * l.temp.Sum() / float64(target.Rows())
}

type crossEntropy struct{ *outLayer }

var quadCost = blas.Binary64(func(x, y float64) float64 {
	res := -y*math.Log(x) - (1-y)*math.Log(1-x)
	if math.IsNaN(res) {
		return 0
	}
	return res
})

// CrossEntropyOutput method appends a cross entropy output layer to the network.
func (n *Network) CrossEntropyOutput(nodes int, a Activation) {
	n.Add(&crossEntropy{newOutLayer(n.batchSize, nodes, a)})
}

func (l *crossEntropy) BackProp(target blas.Matrix, eta float64) blas.Matrix {
	return l.delta.Sub(l.values, target)
}

func (l *crossEntropy) Cost(target blas.Matrix) float64 {
	quadCost.Apply(l.values, target, l.temp)
	return 2 * l.temp.Sum() / float64(target.Rows())
}

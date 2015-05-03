package network

import (
	"github.com/jnb666/deepthought/m32"
	"math"
)

// Layer interface type represents one layer in the network.
type Layer interface {
	FeedForward(in *m32.Matrix) *m32.Matrix
	BackProp(err *m32.Matrix, eta float32) *m32.Matrix
	Weights() *m32.Matrix
	Gradient() *m32.Matrix
	Cost(t *m32.Matrix) float64
}

// Activation interface type represents the activation function and derivative
type Activation interface {
	Func(x, y *m32.Matrix)
	Deriv(x, y *m32.Matrix)
}

var (
	SigmoidActivation = sigmoid{}
	TanhActivation    = tanhFunc{}
)

type sigmoid struct{}

func (a sigmoid) Func(m1, m2 *m32.Matrix) {
	m2.Apply(m1, func(x float32) float32 { return 1 / (1 + exp(-x)) })
}

func (a sigmoid) Deriv(m1, m2 *m32.Matrix) {
	a.Func(m1, m2)
	m2.Apply(m2, func(y float32) float32 { return y * (1 - y) })
}

type tanhFunc struct{}

func (a tanhFunc) Func(m1, m2 *m32.Matrix) {
	m2.Apply(m1, func(x float32) float32 { return tanh(x) })
}

func (a tanhFunc) Deriv(m1, m2 *m32.Matrix) {
	m2.Apply(m1, func(x float32) float32 { y := tanh(x); return 1 - y*y })
}

type inputLayer struct {
	bias     *m32.Matrix // column vector of 1s
	output   *m32.Matrix // return value at each node [samples, nout]
	values   *m32.Matrix // Z matrix of value at each node [samples, nin+1]
	weights  *m32.Matrix // W outgoing weight matrix  [nin+1, nout]
	gradient *m32.Matrix // gradient of weight matrix [nin+1, nout]
}

func newInputLayer(batch, nin, nout int, bias *m32.Matrix) *inputLayer {
	return &inputLayer{
		output:   m32.New(batch, nout),
		values:   m32.New(batch, nin+1),
		weights:  m32.New(nin+1, nout),
		gradient: m32.New(nin+1, nout),
		bias:     bias,
	}
}

// InputLayer method adds a linear input layer to the network.
func (n *Network) InputLayer(nin, nout int) {
	n.Add(newInputLayer(n.batchSize, nin, nout, n.bias))
}

func (l *inputLayer) Weights() *m32.Matrix {
	return l.weights
}

func (l *inputLayer) Gradient() *m32.Matrix {
	return l.gradient
}

func (l *inputLayer) Cost(t *m32.Matrix) float64 {
	panic("no cost for input or hidden layer!")
}

func (l *inputLayer) FeedForward(in *m32.Matrix) *m32.Matrix {
	l.bias.Rows = in.Rows
	l.values.Join(in, l.bias)
	return l.output.Mul(l.values, l.weights)
}

func (l *inputLayer) BackProp(err *m32.Matrix, eta float32) *m32.Matrix {
	etas := -eta / float32(l.values.Rows)
	l.gradient.Mul(err, l.values).Transpose().Scale(etas) // [nout, samples] x [samples, nin+1]
	l.weights.Add(1, l.weights, l.gradient)               // [nin+1, nout]
	return nil
}

type hiddenLayer struct {
	*inputLayer
	activation Activation
	deriv      *m32.Matrix // Fp matrix of derivative of activation fn [nin, samples]
	delta      *m32.Matrix // D matrix of errors at each node [nout, samples]
	wnobias    *m32.Matrix // W matrix without bias weights [nin, nout]
}

// HiddenLayer method adds a new fully connected hidden layer to the network.
func (n *Network) HiddenLayer(nin, nout int, a Activation) {
	layer := &hiddenLayer{
		inputLayer: newInputLayer(n.batchSize, nin, nout, n.bias),
		activation: a,
		deriv:      m32.New(nin, n.batchSize),
		wnobias:    m32.New(nin, nout),
		delta:      m32.New(nin, n.batchSize),
	}
	n.Add(layer)
}

func (l *hiddenLayer) FeedForward(in *m32.Matrix) *m32.Matrix {
	l.bias.Rows = in.Rows
	l.activation.Func(in, l.values)
	l.activation.Deriv(in, l.deriv)
	l.deriv.Transpose()
	l.values.Join(l.values, l.bias)
	return l.output.Mul(l.values, l.weights)
}

func (l *hiddenLayer) BackProp(err *m32.Matrix, eta float32) *m32.Matrix {
	// propagate error backward
	l.wnobias.CopyRows(l.weights, 0, l.weights.Rows-1) // [nin, nout]
	l.delta.Mul(l.wnobias, err)                        // [nin, nout] x [nout, samples]
	l.delta.MulElem(l.delta, l.deriv)                  // [nin, samples]
	// update weights
	etas := -eta / float32(l.values.Rows)
	l.gradient.Mul(err, l.values).Transpose().Scale(etas) // [nout, samples] x [samples, nin+1]
	l.weights.Add(1, l.weights, l.gradient)               // [nin+1, nout]
	return l.delta
}

type outputLayer struct {
	activation Activation
	values     *m32.Matrix // Z matrix of values at each node [samples, nodes]
	delta      *m32.Matrix // D matrix of errors at each node [nodes, samples]
	deriv      *m32.Matrix // Fp matrix of derivative of activation fn [nin, samples]
	temp       *m32.Matrix
}

func newOutputLayer(batch, nodes int, a Activation) *outputLayer {
	return &outputLayer{
		activation: a,
		values:     m32.New(batch, nodes),
		delta:      m32.New(nodes, batch),
		deriv:      m32.New(nodes, batch),
		temp:       m32.New(batch, nodes),
	}
}

func (l *outputLayer) Weights() *m32.Matrix {
	panic("no weights for output layer!")
}

func (l *outputLayer) Gradient() *m32.Matrix {
	panic("no gradient for output layer!")
}

func (l *outputLayer) FeedForward(in *m32.Matrix) *m32.Matrix {
	l.activation.Func(in, l.values)
	l.activation.Deriv(in, l.deriv)
	l.deriv.Transpose()
	return l.values
}

type quadraticOutput struct{ *outputLayer }

// QuadraticOutput method appends a quadratic cost output layer to the network.
func (n *Network) QuadraticOutput(nodes int, a Activation) {
	n.Add(&quadraticOutput{newOutputLayer(n.batchSize, nodes, a)})
}

func (l *quadraticOutput) BackProp(target *m32.Matrix, eta float32) *m32.Matrix {
	l.delta.Add(-1, target, l.values).Transpose()
	return l.delta.MulElem(l.delta, l.deriv)
}

func (l *quadraticOutput) Cost(target *m32.Matrix) float64 {
	return 0.5 * m32.SumDiff2(l.values, target) / float64(target.Rows)
}

type crossEntropy struct{ *outputLayer }

// CrossEntropyOutput method appends a cross entropy output layer to the network.
func (n *Network) CrossEntropyOutput(nodes int, a Activation) {
	n.Add(&crossEntropy{newOutputLayer(n.batchSize, nodes, a)})
}

func (l *crossEntropy) BackProp(target *m32.Matrix, eta float32) *m32.Matrix {
	return l.delta.Add(-1, target, l.values).Transpose()
}

func (l *crossEntropy) Cost(target *m32.Matrix) float64 {
	l.temp.Apply2(l.values, target,
		func(x, y float32) float32 {
			return nanToNum(-y*log(x) - (1-y)*log(1-x))
		})
	return 2 * l.temp.Sum() / float64(target.Rows)
}

// util functions
func exp(x float32) float32 { return float32(math.Exp(float64(x))) }

func tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }

func log(x float32) float32 { return float32(math.Log(float64(x))) }

func nanToNum(x float32) float32 {
	if x != x {
		return 0
	} else {
		return x
	}
}

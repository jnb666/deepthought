package network

import (
	"fmt"
	"github.com/jnb666/deepthought/m32"
	"math"
)

const epsilon = 1e-10

func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// Activation function and derivative
var (
	Activation      = func(x float32) float32 { return (tanh(x) + 1) / 2 }
	ActivationDeriv = func(x float32) float32 { y := tanh(x); return (1 - y*y) / 2 }
)

// Layer interface type represents one layer in the network.
type Layer interface {
	Weights() *m32.Matrix
	FeedForward(input *m32.Matrix) *m32.Matrix
	BackProp(input, target *m32.Matrix, learnRate float32)
	String() string
}

// FCLayer type represents a fully connected layer.
type FCLayer struct {
	weights   *m32.Matrix
	nin, nout int
}

// NewFCLayer function creates a new fully connected layer.
func NewFCLayer(inputs, outputs int) Layer {
	return &FCLayer{
		weights: m32.New(inputs+1, outputs),
		nin:     inputs,
		nout:    outputs,
	}
}

// Weights method returns the weights matrix
func (l *FCLayer) Weights() *m32.Matrix {
	return l.weights
}

// String method returns a printable representation of the layer.
func (l *FCLayer) String() string {
	s := fmt.Sprintf("fully connected layer:\n%s", l.weights)
	return s
}

// FeedForward method evaluates the network layer and returns the output vector for a given input vector.
func (l *FCLayer) FeedForward(input *m32.Matrix) *m32.Matrix {
	return m32.New(input.Rows, l.nout).Mul(input, l.weights).Apply(Activation)
}

// BackProp method performs one back propagation step to tune the weights.
func (l *FCLayer) BackProp(input, target *m32.Matrix, learnRate float32) {
	samples := input.Rows
	// feed forward input
	net := m32.New(samples, l.nout).Mul(input, l.weights)
	output := net.Copy().Apply(Activation)
	// calculate sensitivity
	errorVec := m32.New(samples, l.nout).Add(-1, output, target)
	delta := errorVec.MulElem(errorVec, net.Apply(ActivationDeriv)).Transpose()
	// update weights
	gradient := m32.New(l.nout, l.nin+1).Mul(delta, input).Scale(learnRate / float32(samples))
	l.weights.Add(1, l.weights, gradient.Transpose())
}

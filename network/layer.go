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
	Errors() *m32.Matrix
	FeedForward(input *m32.Matrix) *m32.Matrix
	BackProp(learnRate float32, err *m32.Matrix)
	String() string
}

// FCLayer type represents a fully connected layer.
type FCLayer struct {
	nin, nout int
	weights   *m32.Matrix
	gradient  *m32.Matrix
	input     *m32.Matrix
	delta     *m32.Matrix
	output    *m32.Matrix
	net       *m32.Matrix
}

// NewFCLayer function creates a new fully connected layer.
func NewFCLayer(ninput, noutput, maxSamples int) Layer {
	return &FCLayer{
		nin:      ninput,
		nout:     noutput,
		weights:  m32.New(ninput+1, noutput),
		gradient: m32.New(ninput+1, noutput),
		input:    m32.New(ninput+1, noutput),
		delta:    m32.New(maxSamples, noutput),
		output:   m32.New(maxSamples, noutput),
		net:      m32.New(maxSamples, noutput),
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
	l.input = input
	l.net.Mul(input, l.weights, false)
	l.output.Apply(l.net, Activation)
	return l.output
}

// Errors method returns the output errors matrix
func (l *FCLayer) Errors() *m32.Matrix {
	return l.delta
}

// BackProp method performs one back propagation step to tune the weights.
func (l *FCLayer) BackProp(learnRate float32, err *m32.Matrix) {
	// calculate sensitivity
	l.delta.MulElem(l.delta, l.net.Apply(l.net, ActivationDeriv))
	// update weights
	l.gradient.Mul(l.delta, l.input, true).Scale(learnRate / float32(l.input.Rows)).Transpose()
	l.weights.Add(1, l.weights, l.gradient)
	if err != nil {
		// propagate errors backwards
		err.Mul(l.weights, l.delta, false)
	}
}

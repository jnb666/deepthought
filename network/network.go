package network

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	"math"
	"math/rand"
	"strings"
	"time"
)

const (
	epsilon = 1e-4
)

// Activation type represents the activation function and derivative
type Activation struct {
	Func  blas.UnaryFunction
	Deriv blas.UnaryFunction
}

// Standard activation functions
var (
	NilFunc = Activation{nil, nil}
	Sigmoid = Activation{blas.Unary64(sigmoid), blas.Unary64(dsigmoid)}
	Tanh    = Activation{blas.Unary64(math.Tanh), blas.Unary64(dtanh)}
)

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func dsigmoid(x float64) float64 { y := sigmoid(x); return y * (1 - y) }

func dtanh(x float64) float64 { y := math.Tanh(x); return 1 - y*y }

// Neural network type is an array of layers.
type Network struct {
	nodes      []Layer
	layers     int
	batchSize  int
	classes    blas.Matrix
	bias       blas.Matrix
	out2class  blas.UnaryFunction
	checkEvery int
	checkMax   float64
}

// NewNetwork function initialises a new network, samples is the maximum number of samples, i.e. minibatch size.
func NewNetwork(samples int) *Network {
	return &Network{
		batchSize: samples,
		classes:   blas.New(samples, 1),
		bias:      blas.New(samples, 1).Load(blas.ColMajor, 1),
	}
}

// Add method appends a new layer to the network
func (n *Network) Add(l Layer) {
	n.nodes = append(n.nodes, l)
	n.layers++
}

// String method returns a printable representation of the network.
func (n *Network) String() string {
	str := make([]string, len(n.nodes))
	for i, layer := range n.nodes[:n.layers-1] {
		str[i] = fmt.Sprintf("== Layer %d ==\n%s", i, layer.Weights())
	}
	return strings.Join(str, "\n")
}

// SetRandomWeights method initalises the weights to random values.
// Uses a normal distribution with mean zero and std dev 1/sqrt(num_inputs) for the weights.
// Bias weights are left at zero.
func (n *Network) SetRandomWeights() {
	for _, layer := range n.nodes[:n.layers-1] {
		w := layer.Weights()
		nin, nout := w.Cols(), w.Rows()
		data := make([]float64, nin*nout)
		for i := range data[:(nin-1)*nout] {
			data[i] = rand.NormFloat64() / math.Sqrt(float64(nin-1))
		}
		w.Load(blas.ColMajor, data...)
	}
}

// FeedForward method calculates output from the network given input
func (n *Network) FeedForward(m blas.Matrix) blas.Matrix {
	for _, layer := range n.nodes {
		m = layer.FeedForward(m)
	}
	return m
}

// GetError method calculates the error and classification error given a set of inputs and target outputs.
func (n *Network) GetError(input, target, targetClass blas.Matrix) (totalError, classError float64) {
	output := n.FeedForward(input)
	totalError = n.nodes[n.layers-1].Cost(target) / float64(output.Cols())
	n.out2class.Apply(output, n.classes)
	n.classes.Cmp(n.classes, targetClass, epsilon)
	classError = n.classes.Sum() / float64(output.Rows())
	return
}

// CheckGradient method enables gradient check every nepochs runs with max error of maxError
func (n *Network) CheckGradient(nepochs int, maxError float64) {
	n.checkEvery = nepochs
	n.checkMax = maxError
}

func (n *Network) doCheck(input, target blas.Matrix) (ok bool) {
	output := n.nodes[n.layers-1]
	ok = true
	for i, layer := range n.nodes[:n.layers-1] {
		w := layer.Weights()
		weight := w.Data(blas.ColMajor)
		wsave := append([]float64{}, weight...)
		grads := layer.Gradient().Data(blas.ColMajor)
		diffs := make([]float64, w.Size())
		projs := make([]float64, w.Size())
		maxDiff := 0.0
		for i, val := range weight {
			weight[i] = val - epsilon
			w.Load(blas.ColMajor, weight...)
			n.FeedForward(input)
			cost1 := output.Cost(target)
			weight[i] = val + epsilon
			w.Load(blas.ColMajor, weight...)
			n.FeedForward(input)
			cost2 := output.Cost(target)
			projs[i] = (cost1 - cost2) / (2 * epsilon)
			diffs[i] = grads[i] - projs[i]
			maxDiff = math.Max(maxDiff, math.Abs(diffs[i]))
		}
		fmt.Printf("LAYER %d : max diff=%.8f\n", i, maxDiff)
		if maxDiff > n.checkMax {
			layer.Gradient().SetFormat("%9.6f")
			proj := blas.New(w.Rows(), w.Cols()).Load(blas.ColMajor, projs...)
			proj.SetFormat("%9.6f")
			diff := blas.New(w.Rows(), w.Cols()).Load(blas.ColMajor, diffs...)
			diff.SetFormat("%9.6f")
			fmt.Printf("*** WARNING *** GRADIENTS LOOK WRONG!\n%s\n\n%s\n\n%s\n",
				layer.Gradient(), proj, diff)
			ok = false
		}
		w.Load(blas.ColMajor, wsave...)
	}
	return
}

// Train method trains the network on the given training set and updates the stats.
// stop callback function returns true if we should terminate the run.
func (n *Network) Train(d data.Dataset, learnRate float64, s *Stats, stop func(*Stats) bool) int {
	n.out2class = d.OutputToClass
	s.StartRun()
	for {
		n.FeedForward(d.Train.Input)
		delta := d.Train.Output
		for i := n.layers - 1; i >= 0; i-- {
			delta = n.nodes[i].BackProp(delta, learnRate)
		}
		if n.checkEvery > 0 && s.Epoch%n.checkEvery == 0 {
			n.doCheck(d.Train.Input, d.Train.Output)
		}
		s.Update(n, d)
		if stop(s) {
			break
		}
		s.Epoch++
	}
	s.EndRun()
	return s.Epoch
}

// SeedRandom function sets the random seed, or seeds using time if input is zero
func SeedRandom(seed int64) {
	if seed == 0 {
		seed = time.Now().UTC().UnixNano()
	}
	fmt.Println("random seed =", seed)
	rand.Seed(seed)
}

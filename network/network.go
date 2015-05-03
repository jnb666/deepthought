package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/m32"
	"math"
	"math/rand"
	"strings"
	"time"
)

const (
	epsilon = 1e-4
)

// Neural network type is an array of layers.
type Network struct {
	nodes      []Layer
	delta      []*m32.Matrix
	layers     int
	batchSize  int
	classes    *m32.Matrix
	bias       *m32.Matrix
	out2class  func(a, b *m32.Matrix)
	checkEvery int
	checkMax   float64
}

// NewNetwork function initialises a new network, samples is the maximum number of samples, i.e. minibatch size.
func NewNetwork(samples int) *Network {
	return &Network{
		batchSize: samples,
		classes:   m32.New(samples, 1),
		bias:      m32.New(samples, 1).Load(m32.ColMajor, 1),
	}
}

// Add method appends a new layer to the network
func (n *Network) Add(l Layer) {
	n.nodes = append(n.nodes, l)
	n.delta = append(n.delta, nil)
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
		nin, nout := w.Rows-1, w.Cols
		data := make([]float32, w.Rows*w.Cols)
		for i := range data[:nin*nout] {
			data[i] = float32(rand.NormFloat64() / math.Sqrt(float64(nin)))
		}
		w.Load(m32.RowMajor, data...)
	}
}

// FeedForward method calculates output from the network given input
func (n *Network) FeedForward(m *m32.Matrix) *m32.Matrix {
	for _, layer := range n.nodes {
		m = layer.FeedForward(m)
	}
	return m
}

// BackProp method performs backpropagation of the errors for each layer and updates the weights
func (n *Network) BackProp(target *m32.Matrix, eta float32) {
	n.delta[n.layers-1] = n.nodes[n.layers-1].BackProp(target, eta)
	for i := n.layers - 2; i >= 0; i-- {
		n.delta[i] = n.nodes[i].BackProp(n.delta[i+1], eta)
	}
}

// GetError method calculates the error and classification error given a set of inputs and target outputs.
func (n *Network) GetError(input, target, targetClass *m32.Matrix) (totalError, classError float64) {
	output := n.FeedForward(input)
	totalError = float64(m32.SumDiff2(target, output)) / float64(input.Rows*output.Cols)
	n.out2class(output, n.classes)
	classError = float64(m32.CountDiff(n.classes, targetClass, epsilon)) / float64(input.Rows)
	return
}

// CheckGradient method enables gradient check every nepochs runs with max error of maxError
func (n *Network) CheckGradient(nepochs int, maxError float64) {
	n.checkEvery = nepochs
	n.checkMax = maxError
}

func (n *Network) doCheck(input, target *m32.Matrix) (ok bool) {
	output := n.nodes[n.layers-1]
	ok = true
	for i, layer := range n.nodes[:n.layers-1] {
		w := layer.Weights()
		weight := w.Data()
		grads := layer.Gradient().Data()
		diffs := make([]float32, w.Rows*w.Cols)
		projs := make([]float32, w.Rows*w.Cols)
		maxDiff := 0.0
		for i, val := range weight {
			weight[i] = val - epsilon
			n.FeedForward(input)
			cost1 := output.Cost(target)
			weight[i] = val + epsilon
			n.FeedForward(input)
			cost2 := output.Cost(target)
			g1 := float64(grads[i])
			g2 := -(cost2 - cost1) / (2.0 * epsilon)
			dg := math.Abs(g1 - g2)
			maxDiff = math.Max(maxDiff, dg)
			projs[i] = float32(g2)
			diffs[i] = float32(dg)
			weight[i] = val
		}
		fmt.Printf("LAYER %d : max diff=%.8f\n", i, maxDiff)
		if maxDiff > n.checkMax {
			layer.Gradient().SetFormat("%9.6f")
			proj := m32.New(w.Rows, w.Cols).Load(m32.ColMajor, projs...)
			proj.SetFormat("%9.6f")
			diff := m32.New(w.Rows, w.Cols).Load(m32.ColMajor, diffs...)
			diff.SetFormat("%9.6f")
			fmt.Printf("*** WARNING *** GRADIENTS LOOK WRONG!\n%s\n\n%s\n\n%s\n",
				layer.Gradient(), proj, diff)
			ok = false
		}
	}
	return
}

// Train method trains the network on the given training set and updates the stats.
// stop callback function returns true if we should terminate the run.
func (n *Network) Train(d data.Dataset, learnRate float32, s *Stats, stop func(*Stats) bool) int {
	n.out2class = d.OutputToClass
	s.StartRun()
	for {
		n.FeedForward(d.Train.Input)
		n.BackProp(d.Train.Output, learnRate)
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

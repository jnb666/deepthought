package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/m32"
	"math/rand"
	"strings"
	"time"
)

const epsilon = 1e-10

// Neural network type is an array of layers.
type Network struct {
	nodes     []Layer
	delta     []*m32.Matrix
	layers    int
	batchSize int
	classes   *m32.Matrix
	bias      *m32.Matrix
	out2class func(a, b *m32.Matrix)
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

// SetRandomWeights method initalises the weights to random values from -max to +max.
func (n *Network) SetRandomWeights(max float32) {
	for _, l := range n.nodes[:n.layers-1] {
		l.Weights().Random(-max, max)
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
func (n *Network) GetError(input, target, targetClass *m32.Matrix) (totalError, classError float32) {
	output := n.FeedForward(input)
	totalError = m32.SumDiff2(target, output) / float32(input.Rows*output.Cols)
	n.out2class(output, n.classes)
	classError = m32.CountDiff(n.classes, targetClass, epsilon) / float32(input.Rows)
	return
}

// Train method trains the network on the given training set and updates the stats.
// stop callback function returns true if we should terminate the run.
func (n *Network) Train(d data.Dataset, learnRate float32, s *Stats, stop func(int) bool) int {
	n.out2class = d.OutputToClass
	s.Clear()
	start := time.Now()
	for {
		n.FeedForward(d.Train.Input)
		n.BackProp(d.Train.Output, learnRate)
		s.Update(n, d)
		if stop(s.Epoch) {
			break
		}
		s.Epoch++
	}
	s.NumEpochs.Push(float64(s.Epoch + 1))
	s.RunTime.Push(time.Since(start).Seconds())
	test := s.Test
	if test.Error.Len() == 0 {
		test = s.Train
	}
	s.RegError.Push(test.Error.Last())
	s.ClsError.Push(test.ClassError.Last())
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

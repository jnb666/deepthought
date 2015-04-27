package network

import (
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/m32"
	"strings"
)

// Neural network type is an array of layers.
type Network struct {
	Nodes []Layer
}

// NewNetwork function initialises a new network
func NewNetwork() *Network {
	return &Network{Nodes: []Layer{}}
}

// Add method appends a new layer to the network
func (n *Network) Add(l Layer) {
	n.Nodes = append(n.Nodes, l)
}

// String method returns a printable representation of the network.
func (n *Network) String() string {
	str := make([]string, len(n.Nodes))
	for i, layer := range n.Nodes {
		str[i] = layer.String()
	}
	return strings.Join(str, "\n")
}

// SetRandomWeights method initalises the weights to random values from -max to +max.
func (n *Network) SetRandomWeights(max float32) {
	for _, l := range n.Nodes {
		l.Weights().Random(-max, max)
	}
}

// Run method calculates output from the network given input
func (n *Network) Run(input *m32.Matrix) *m32.Matrix {
	output := n.Nodes[0].FeedForward(input)
	for _, layer := range n.Nodes[1:] {
		output = layer.FeedForward(output)
	}
	return output
}

// GetError method calculates the error and classification error given a set of inputs and target outputs.
func (n *Network) GetError(input, target, targetClass *m32.Matrix) (totalError, classError float32) {
	output := n.Run(input)
	totalError = m32.SumDiff2(target, output) / float32(input.Rows*output.Cols)
	classes := m32.New(output.Rows, 1).MaxCol(output)
	classError = m32.CountDiff(classes, targetClass, epsilon) / float32(input.Rows)
	return
}

// Train method trains the network on the given training set and updates the stats.
// stop callback function returns true if we should terminate the run.
func (n *Network) Train(t data.Dataset, learnRate float32, s *Stats, stop func(int) bool) int {
	// reset stats
	s.Clear()
	for {
		// update weights
		// TODO: add hidden nodes
		layer := n.Nodes[0]
		layer.BackProp(t.Train.Input, t.Train.Output, learnRate)

		// update stats
		s.Update(n, t)

		// next epoch
		if stop(s.Epoch) {
			return s.Epoch
		}
		s.Epoch++
	}
}

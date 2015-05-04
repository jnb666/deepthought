package network

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"testing"
)

const maxEpoch = 200

func init() {
	blas.Init(blas.Native64)
}

func createNetwork(batchSize int) (net *Network, s data.Dataset, err error) {
	if s, err = data.Load("iris", batchSize); err != nil {
		return
	}
	net = NewNetwork(s.MaxSamples)
	net.AddLayer(s.NumInputs, s.NumOutputs, NilFunc)
	net.QuadraticOutput(s.NumOutputs, Sigmoid)
	net.SetRandomWeights()
	return
}

func TestNetwork(t *testing.T) {
	net, _, err := createNetwork(10)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(net)
}

func TestFeedForward(t *testing.T) {
	net, d, err := createNetwork(10)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(net)
	t.Logf("input:\n%s\n", d.Test.Input)
	output := net.FeedForward(d.Test.Input)
	t.Logf("output:\n%s\n", output)
}

func TestTrain(t *testing.T) {
	net, d, err := createNetwork(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("read %d test %d train and %d validation samples: max=%d\n",
		d.Test.NumSamples, d.Train.NumSamples, d.Valid.NumSamples, d.MaxSamples)
	t.Log(net)
	stats := NewStats(maxEpoch, 1)
	epochs := net.Train(d, 10, stats, func(s *Stats) bool {
		done := s.Epoch >= maxEpoch || s.Valid.Error.Last() < 0.05
		if s.Epoch%10 == 0 || done {
			t.Log(s)
		}
		return done
	})
	t.Logf("completed after %d epochs\n", epochs)
	if epochs == maxEpoch {
		t.Error("training failed to hit threshold!")
	}
	t.Log(net)
}

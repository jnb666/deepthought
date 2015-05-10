package network

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"testing"
)

const maxEpoch = 200

func init() {
	//Debug = true
	//Init(blas.Native32)
	Init(blas.OpenCL32)
}

func createNetwork(samples int) (net *Network, s *data.Dataset, err error) {
	if s, err = data.Load("iris", samples, 0); err != nil {
		return
	}
	net = NewNetwork(s.MaxSamples)
	net.AddLayer(s.NumInputs, s.NumOutputs, Linear)
	net.QuadraticOutput(s.NumOutputs, Sigmoid)
	net.SetRandomWeights()
	return
}

func TestNetwork(t *testing.T) {
	net, _, err := createNetwork(10)
	if err != nil {
		t.Fatal(err)
	}
	defer net.Release()
	t.Log(net)
}

func TestFeedForward(t *testing.T) {
	net, d, err := createNetwork(10)
	if err != nil {
		t.Fatal(err)
	}
	defer net.Release()
	t.Log(net)
	t.Logf("input:\n%s\n", d.Test.Input[0])
	output := net.FeedForward(d.Test.Input[0])
	t.Logf("output:\n%s\n", output)
	if output.Rows() != 10 || output.Cols() != 3 {
		t.Error("output is wrong size!")
	}
}

func TestTrain(t *testing.T) {
	net, d, err := createNetwork(0)
	if err != nil {
		t.Fatal(err)
	}
	defer net.Release()
	t.Logf("read %d test %d train and %d validation samples: max=%d\n",
		d.Test.NumSamples, d.Train.NumSamples, d.Valid.NumSamples, d.MaxSamples)
	t.Log(net)
	stats := NewStats(maxEpoch, 1)
	epochs := net.Train(d, 10, stats, func(s *Stats) bool {
		done := s.Epoch >= maxEpoch || s.Valid.Error.Last() < 0.1
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

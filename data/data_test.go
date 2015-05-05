package data_test

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	_ "github.com/jnb666/deepthought/data/mnist"
	_ "github.com/jnb666/deepthought/data/xor"
	"testing"
)

func init() {
	blas.Init(blas.Native64)
}

func load(t *testing.T, name string, max, batch int) *data.Dataset {
	s, err := data.Load(name, max, batch)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestIris(t *testing.T) {
	s := load(t, "iris", 10, 0)
	t.Log(s.Test)
}

func TestXor(t *testing.T) {
	s := load(t, "xor", 0, 0)
	t.Log(s.Train)
}

func TestMNIST(t *testing.T) {
	s := load(t, "mnist", 0, 100)
	//t.Log(s.Train)
	nbatch := len(s.Train.Input)
	if nbatch != 500 {
		t.Errorf("incorrect no. of batches: got %d", nbatch)
	}
	if s.Train.NumSamples != nbatch*100 {
		t.Errorf("incorrect number of samples: got %d", s.Train.NumSamples)
	}
}

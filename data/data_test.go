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
	blas.Init(blas.Native32)
}

func load(t *testing.T, name string, max int) *data.Dataset {
	s, err := data.Load(name, max)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestIris(t *testing.T) {
	s := load(t, "iris", 10)
	t.Log(s.Test)
}

func TestXor(t *testing.T) {
	s := load(t, "xor", 0)
	t.Log(s.Train)
}

func TestMNIST(t *testing.T) {
	s := load(t, "mnist", 0)
	entry := 2
	img := s.Train.Input.Row(entry, entry+1).Reshape(28, 28, true)
	in := blas.New(28, 28).Copy(img, nil)
	in.SetFormat("%c")
	t.Logf("input:\n%s\n", in)
	out := s.Train.Output.Row(entry, entry+1)
	out.SetFormat("%3.0f")
	t.Logf("output:\n%s\n", out)
	class := s.Train.Classes.Row(entry, entry+1)
	class.SetFormat("%3.0f")
	t.Logf("class:\n%s\n", class)
	if s.Train.NumSamples != 50000 || s.Test.NumSamples != 10000 || s.Valid.NumSamples != 10000 {
		t.Errorf("incorrect number of samples: got %d %d %d", s.Train.NumSamples, s.Test.NumSamples, s.Valid.NumSamples)
	}
}

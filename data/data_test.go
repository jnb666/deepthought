package data_test

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	_ "github.com/jnb666/deepthought/data/xor"
	"testing"
)

func init() {
	blas.Init(blas.Native64)
}

func load(t *testing.T, name string, max int) {
	s, err := data.Load(name, max)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("input:\n%s\n", s.Train.Input)
	t.Logf("output:\n%s\n", s.Train.Output)
	t.Logf("classes:\n%s\n", s.Train.Classes)
}

func TestIris(t *testing.T) {
	load(t, "iris", 10)
}

func TestXor(t *testing.T) {
	load(t, "xor", 0)
}

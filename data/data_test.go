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

func TestBatch(t *testing.T) {
	s := load(t, "iris", 0, 10)
	t.Log(s.Train)
	if len(s.Train.Input) != 7 {
		t.Error("incorrect no. of batches!")
	}
}

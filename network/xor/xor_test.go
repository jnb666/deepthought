package xor

import (
	"github.com/jnb666/deepthought/blas"
	"testing"
)

func init() {
	blas.Init(blas.Native32)
}

func TestXor(t *testing.T) {
	s, err := Loader{}.Load(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(s.Train)
}

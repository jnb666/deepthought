package iris

import (
	"github.com/jnb666/deepthought/blas"
	"testing"
)

func init() {
	blas.Init(blas.Native32)
}

func TestIris(t *testing.T) {
	s, err := Loader{}.Load(10)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(s.Test)
}

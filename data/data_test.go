package data_test

import (
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"testing"
)

func TestLayer(t *testing.T) {
	s, err := data.Load("iris", 10)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("input:\n%s\n", s.Test.Input)
	t.Logf("output:\n%s\n", s.Test.Output)
	t.Logf("classes:\n%s\n", s.Test.Classes)
}

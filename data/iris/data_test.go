package iris

import (
	"testing"
)

func TestLayer(t *testing.T) {
	s, err := Load(10)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("input:\n%s\n", s.Test.Input)
	t.Logf("output:\n%s\n", s.Test.Output)
	t.Logf("classes:\n%s\n", s.Test.Classes)
}

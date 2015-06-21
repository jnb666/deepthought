package mnist

import (
	"github.com/jnb666/deepthought/blas"
	"testing"
	"time"
)

var l = &Loader{}

func init() {
	blas.Init(blas.OpenCL32)
}

func getImage(array blas.Matrix, ix int) blas.Matrix {
	data := array.Row(ix, ix+1).Data(blas.RowMajor)
	img := blas.New(size, size).Load(blas.RowMajor, data...)
	img.SetFormat("%c")
	return img
}

func TestMNIST(t *testing.T) {
	s, err := l.Load(0)
	if err != nil {
		t.Fatal(err)
	}
	entry := 2
	in := getImage(s.Train.Input, entry)
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

func TestDistort(t *testing.T) {
	nimage := 50000
	batch := 250
	verbose := false
	mask := Scale | Rotate | Elastic
	s, err := l.Load(nimage)
	if err != nil {
		t.Fatal(err)
	}
	var input blas.Matrix
	output := blas.New(batch, size*size)
	start := time.Now()
	for i := 0; i < s.Train.NumSamples/batch; i++ {
		input = s.Train.Input.Row(i*batch, (i+1)*batch)
		l.Distort(input, output, mask, 0.1)
		if verbose {
			t.Logf("input \n%s\n", getImage(input, 0))
			t.Logf("output\n%s\n", getImage(output, 0))
		}
	}
	blas.Sync()
	t.Log("images=", nimage, " runtime=", time.Since(start))
	output.Release()
	l.Release()
}

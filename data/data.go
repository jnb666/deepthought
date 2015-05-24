// Package data contains routines to load data sets into matrices for neural network processing.
package data

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"os"
)

// Register map of all loaders which are available
var Register = map[string]Loader{}

// Loader interface is used to load a new dataset
type Loader interface {
	Load(samples int) (*Dataset, error)
}

// Dataset type represents a set of test, training and validation data
type Dataset struct {
	OutputToClass blas.UnaryFunction
	Test          *Data
	Train         *Data
	Valid         *Data
	NumInputs     int
	NumOutputs    int
	MaxSamples    int
}

// Data type represents a set of test or training data.
type Data struct {
	Input      blas.Matrix
	Output     blas.Matrix
	Classes    blas.Matrix
	NumSamples int
}

func (d *Data) String() string {
	return fmt.Sprintf("total samples:%d\ninput:\n%s\noutput:\n%s\nclasses:\n%s",
		d.NumSamples, d.Input, d.Output, d.Classes)
}

// Load function loads and returns a new dataset.
func Load(name string, samples int) (s *Dataset, err error) {
	if loader, ok := Register[name]; ok {
		s, err = loader.Load(samples)
	} else {
		err = fmt.Errorf("Load: unknown dataset name %s\n", name)
	}
	return
}

// Release method frees any allocated resources
func (d *Dataset) Release() {
	if d.Test != nil {
		d.Test.Release()
	}
	if d.Train != nil {
		d.Train.Release()
	}
	if d.Valid != nil {
		d.Valid.Release()
	}
}

func (d *Data) Release() {
	d.Input.Release()
	d.Output.Release()
	d.Classes.Release()
}

// LoadFile function reads a dataset from a text file.
// samples is maxiumum number of records to load from each dataset if non-zero.
func LoadFile(filename string, samples int, out2class blas.UnaryFunction) (d *Data, nin, nout int, err error) {
	var file *os.File
	if file, err = os.Open(filename); err != nil {
		return
	}
	// header has no. of inputs, no. of outputs and count of samples
	var rows int
	if _, err = fmt.Fscanf(file, "%d %d %d\n", &nin, &nout, &rows); err != nil {
		err = fmt.Errorf("error reading header: %s", err)
		return
	}
	cols := nin + nout
	if samples > 0 && rows > samples {
		rows = samples
	}
	//fmt.Printf("reading from %s: rows=%d, cols=%d, batches=%d x %d\n", filename, rows, cols, nBatch, batchSize)
	d = new(Data)
	d.NumSamples = rows
	// read data
	buf := make([]float64, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			if _, err = fmt.Fscan(file, &buf[col*rows+row]); err != nil {
				err = fmt.Errorf("error reading data value: %s", err)
				return
			}
		}
	}
	// format as matrix
	d.Input = blas.New(rows, nin).Load(blas.ColMajor, buf...)
	d.Output = blas.New(rows, nout).Load(blas.ColMajor, buf[rows*nin:]...)
	d.Classes = blas.New(rows, 1)
	out2class.Apply(d.Output, d.Classes)
	return
}

package data

import (
	"fmt"
	"github.com/jnb666/deepthought/m32"
	"os"
)

// Dataset type represents a set of test, training and validation data
type Dataset struct {
	Test       Data
	Train      Data
	Valid      Data
	NumInputs  int
	NumOutputs int
}

// Data type represents a set of test or training data.
// An additional column is added to the input to represent the bias nodes.
type Data struct {
	Input      *m32.Matrix
	Output     *m32.Matrix
	Classes    *m32.Matrix
	NumSamples int
}

// Load function reads a dataset from a text file.
// samples is maxiumum number of records to load from each dataset if non-zero.
func Load(filename string, samples int) (d Data, nin, nout int, err error) {
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
	if samples > 0 && rows > samples {
		rows = samples
	}
	cols := 1 + nin + nout
	// read data
	buf := make([]float32, rows*cols)
	for row := 0; row < rows; row++ {
		buf[row] = 1
	}
	for row := 0; row < rows; row++ {
		for col := 1; col < cols; col++ {
			if _, err = fmt.Fscan(file, &buf[col*rows+row]); err != nil {
				err = fmt.Errorf("error reading data value: %s", err)
				return
			}
		}
	}
	// format as matrix
	d.NumSamples = rows
	d.Input = m32.New(rows, nin+1).Load(m32.ColMajor, buf...)
	d.Output = m32.New(rows, nout).Load(m32.ColMajor, buf[rows*(nin+1):]...)
	d.Classes = m32.New(rows, 1).MaxCol(d.Output)
	return
}

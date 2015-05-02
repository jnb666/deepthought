package data

import (
	"fmt"
	"github.com/jnb666/deepthought/m32"
	"os"
)

// Register map of all loaders which are available
var Register = map[string]Loader{}

// Loader interface is used to load a new dataset
type Loader interface {
	Load(samples int) (Dataset, error)
}

// Dataset type represents a set of test, training and validation data
type Dataset struct {
	Test          *Data
	Train         *Data
	Valid         *Data
	NumInputs     int
	NumOutputs    int
	MaxSamples    int
	OutputToClass func(a, b *m32.Matrix)
}

// Data type represents a set of test or training data.
// An additional column is added to the input to represent the bias nodes.
type Data struct {
	Input      *m32.Matrix
	Output     *m32.Matrix
	Classes    *m32.Matrix
	NumSamples int
}

func (d *Data) String() string {
	return fmt.Sprintf("input:\n%s\noutput:\n%s\nclasses:\n%s", d.Input, d.Output, d.Classes)
}

// Load function loads and returns a new dataset.
func Load(name string, samples int) (s Dataset, err error) {
	if loader, ok := Register[name]; ok {
		s, err = loader.Load(samples)
	} else {
		err = fmt.Errorf("Load: unknown dataset name %s\n", name)
	}
	return
}

// LoadFile function reads a dataset from a text file.
// samples is maxiumum number of records to load from each dataset if non-zero.
func LoadFile(filename string, samples int, out2class func(a, b *m32.Matrix)) (d *Data, nin, nout int, err error) {
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
	cols := nin + nout
	// read data
	buf := make([]float32, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			if _, err = fmt.Fscan(file, &buf[col*rows+row]); err != nil {
				err = fmt.Errorf("error reading data value: %s", err)
				return
			}
		}
	}
	// format as matrix
	d = new(Data)
	d.NumSamples = rows
	d.Input = m32.New(rows, nin).Load(m32.ColMajor, buf...)
	d.Output = m32.New(rows, nout).Load(m32.ColMajor, buf[rows*nin:]...)
	d.Classes = m32.New(rows, 1)
	out2class(d.Output, d.Classes)
	return
}

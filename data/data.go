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
	Load(samples, batchSize int) (*Dataset, error)
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
// Each data set is split into batches of BatchSize entries if this is non-zero.
type Data struct {
	Input      []blas.Matrix
	Output     []blas.Matrix
	Classes    []blas.Matrix
	NumSamples int
}

func (d *Data) String() string {
	return fmt.Sprintf("total samples:%d batches=%d\ninput:\n%s\noutput:\n%s\nclasses:\n%s",
		d.NumSamples, len(d.Input), d.Input[0], d.Output[0], d.Classes[0])
}

// Load function loads and returns a new dataset.
func Load(name string, samples, batchSize int) (s *Dataset, err error) {
	if loader, ok := Register[name]; ok {
		s, err = loader.Load(samples, batchSize)
	} else {
		err = fmt.Errorf("Load: unknown dataset name %s\n", name)
	}
	return
}

// LoadFile function reads a dataset from a text file.
// samples is maxiumum number of records to load from each dataset if non-zero.
func LoadFile(filename string, samples, batchSize int, out2class blas.UnaryFunction) (d *Data, nin, nout int, err error) {
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
	if batchSize == 0 {
		batchSize = rows
	}
	nBatch := rows / batchSize
	//fmt.Printf("reading from %s: rows=%d, cols=%d, batches=%d x %d\n", filename, rows, cols, nBatch, batchSize)
	d = new(Data)
	d.NumSamples = nBatch * batchSize
	d.Input = make([]blas.Matrix, nBatch)
	d.Output = make([]blas.Matrix, nBatch)
	d.Classes = make([]blas.Matrix, nBatch)
	for i := range d.Input {
		// read batch of data
		buf := make([]float64, batchSize*cols)
		for row := 0; row < batchSize; row++ {
			for col := 0; col < cols; col++ {
				if _, err = fmt.Fscan(file, &buf[col*batchSize+row]); err != nil {
					err = fmt.Errorf("error reading data value: %s", err)
					return
				}
			}
		}
		// format as matrix
		d.Input[i] = blas.New(batchSize, nin).Load(blas.ColMajor, buf...)
		d.Output[i] = blas.New(batchSize, nout).Load(blas.ColMajor, buf[batchSize*nin:]...)
		d.Classes[i] = blas.New(batchSize, 1)
		out2class.Apply(d.Output[i], d.Classes[i])
	}
	return
}

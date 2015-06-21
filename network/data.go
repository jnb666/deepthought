package network

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"os"
)

var register = map[string]Loader{}

// Distortion type is used for a bitmask of supported distortions
type Distortion struct {
	Mask int
	Name string
}

// Loader interface is used to load a new dataset
type Loader interface {
	Load(samples int) (*Dataset, error)
	Config() *Config
	CreateNetwork(cfg *Config, d *Dataset) *Network
	DistortTypes() []Distortion
	Distort(in, out blas.Matrix, mask int, severity float64)
	Release()
}

// Register function is called on initialisation to make a new dataset available.
func Register(name string, l Loader) {
	register[name] = l
}

// GetLoader function looks up a loader by name.
func GetLoader(name string) (l Loader, ok bool) {
	l, ok = register[name]
	return
}

// Dataset type represents a set of test, training and validation data
type Dataset struct {
	Load          Loader
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
	if d.Load != nil {
		d.Load.Release()
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

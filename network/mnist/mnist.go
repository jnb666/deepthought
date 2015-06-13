// Package mnist loads the MNist dataset of handwritten digits.
package mnist

import (
	"encoding/binary"
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
	"math"
	"os"
)

const (
	// Data directory
	base        = "/home/john/go/src/github.com/jnb666/deepthought/network/mnist/"
	trainImages = "train-images-idx3-ubyte"
	trainLabels = "train-labels-idx1-ubyte"
	testImages  = "t10k-images-idx3-ubyte"
	testLabels  = "t10k-labels-idx1-ubyte"
	numOutputs  = 10
	trainMax    = 50000
	testMax     = 10000
)

// register dataset when module is imported
func init() {
	network.Register("mnist", Loader{})
	network.Register("mnist2", Loader2{})
}

// classification function
type Classify struct{}

func (Classify) Apply(out, class blas.Matrix) blas.Matrix { return class.MaxCol(out) }

// default configuration
type Loader struct{}

func (Loader) Config() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  50,
		BatchSize: 100,
		LearnRate: 3.0,
		Threshold: 0.0067,
		LogEvery:  1,
		Sampler:   "random",
	}
}

func (Loader) CreateNetwork(cfg *network.Config, d *network.Dataset) *network.Network {
	hiddenNodes := 30
	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with quadratic cost and sigmoid activation\n",
		d.NumInputs, hiddenNodes, d.NumOutputs)
	net := network.New(cfg.BatchSize, d.OutputToClass)
	net.AddLayer(dims(d.NumInputs), hiddenNodes, network.Linear)
	net.AddLayer([]int{hiddenNodes}, d.NumOutputs, network.Sigmoid)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	return net
}

// Alternative configuration
type Loader2 struct{ Loader }

func (Loader2) Config() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  100,
		BatchSize: 250,
		LearnRate: 0.5,
		Momentum:  0.7,
		StopAfter: 8,
		LogEvery:  5,
		Sampler:   "random",
	}
}

func (Loader2) CreateNetwork(cfg *network.Config, d *network.Dataset) *network.Network {
	hiddenNodes := 400
	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with cross entropy cost and relu activation\n",
		d.NumInputs, hiddenNodes, d.NumOutputs)
	net := network.New(cfg.BatchSize, d.OutputToClass)
	net.AddLayer(dims(d.NumInputs), hiddenNodes, network.Linear)
	net.AddLayer(dims(hiddenNodes), d.NumOutputs, network.Relu)
	net.AddCrossEntropyOutput(d.NumOutputs)
	return net
}

func dims(n int) []int {
	size := int(math.Sqrt(float64(n)))
	if size*size == n {
		return []int{size, size}
	}
	return []int{n}
}

// rescale intensity to value
func rescale(x uint8) float64 { return float64(x) / 255.0 }

// Load function loads and returns the iris dataset.
func (Loader) Load(samples int) (*network.Dataset, error) {
	s := new(network.Dataset)
	s.OutputToClass = Classify{}

	// training set
	r := newReader(trainLabels, trainImages)
	s.Train = r.read(samples, trainMax)
	s.NumInputs = int(r.image.Width * r.image.Height)
	s.NumOutputs = numOutputs
	s.MaxSamples = s.Train.NumSamples

	// validation set
	r.seek(trainMax)
	s.Valid = r.read(samples, testMax)
	r.close()
	if r.err != nil {
		return s, r.err
	}

	// test set
	r = newReader(testLabels, testImages)
	s.Test = r.read(samples, testMax)
	r.close()
	return s, r.err
}

// imageReader to read from binary file of images and labels
type imageReader struct {
	flab  *os.File
	fimg  *os.File
	label struct{ Magic, Num uint32 }
	image struct{ Magic, Num, Width, Height uint32 }
	err   error
}

// open files and read header info
func newReader(labelFile, imageFile string) (r *imageReader) {
	r = new(imageReader)
	if r.flab, r.err = os.Open(base + labelFile); r.err != nil {
		return
	}
	if r.err = binary.Read(r.flab, binary.BigEndian, &r.label); r.err != nil {
		return
	}
	if r.fimg, r.err = os.Open(base + imageFile); r.err != nil {
		return
	}
	if r.err = binary.Read(r.fimg, binary.BigEndian, &r.image); r.err != nil {
		return
	}
	return
}

// close the files
func (r *imageReader) close() {
	if r.err == nil {
		r.flab.Close()
		r.fimg.Close()
	}
}

// seek to nth image in the file
func (r *imageReader) seek(n int) {
	if r.err != nil {
		return
	}
	offset := int64(n) + 8
	if _, r.err = r.flab.Seek(offset, 0); r.err != nil {
		return
	}
	offset = int64(n)*int64(r.image.Width*r.image.Height) + 16
	if _, r.err = r.fimg.Seek(offset, 0); r.err != nil {
		return
	}
}

// read an entire dataset
func (r *imageReader) read(samples, maxImages int) *network.Data {
	d := new(network.Data)
	if samples == 0 || samples > maxImages {
		samples = maxImages
	}
	d.Input, d.Output, d.Classes = r.readBatch(samples)
	d.NumSamples = samples
	return d
}

// read one batch of n images and associated labels
func (r *imageReader) readBatch(n int) (in, out, class blas.Matrix) {
	labels := make([]byte, n)
	if _, r.err = r.flab.Read(labels); r.err != nil {
		return
	}
	size := int(r.image.Width) // assume square
	image := make([]byte, size*size)
	idata := make([]float64, n*size*size)
	odata := make([]float64, n*numOutputs)
	cdata := make([]float64, n)
	for i := 0; i < n; i++ {
		if _, r.err = r.fimg.Read(image); r.err != nil {
			return
		}
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				idata[i*size*size+y*size+x] = rescale(image[y*size+x])
			}
		}
		for j := 0; j < numOutputs; j++ {
			if int(labels[i]) == j {
				odata[i*numOutputs+j] = 1
			}
			cdata[i] = float64(labels[i])
		}
	}
	in = blas.New(n, size*size).Load(blas.RowMajor, idata...)
	out = blas.New(n, numOutputs).Load(blas.RowMajor, odata...)
	class = blas.New(n, 1).Load(blas.RowMajor, cdata...)
	return
}

package mnist

import (
	"encoding/binary"
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	"os"
)

const (
	// Data directory
	base        = "/home/john/go/src/github.com/jnb666/deepthought/data/mnist/"
	trainImages = "train-images-idx3-ubyte"
	trainLabels = "train-labels-idx1-ubyte"
	testImages  = "t10k-images-idx3-ubyte"
	testLabels  = "t10k-labels-idx1-ubyte"
	numOutputs  = 10
	trainMax    = 50000
	testMax     = 10000
)

var Debug = false

// register dataset when module is imported
func init() {
	data.Register["mnist"] = loader{}
}

// classification function
type classify struct{}

func (classify) Apply(out, class blas.Matrix) blas.Matrix { return class.MaxCol(out) }

type loader struct{}

// rescale intensity to value
func rescale(x uint8) float64 { return float64(x) / 255 }

// Load function loads and returns the iris dataset.
// samples is maxiumum number of records to load from each dataset if non-zero.
func (loader) Load(samples, batchSize int) (*data.Dataset, error) {
	s := new(data.Dataset)
	s.OutputToClass = classify{}
	if batchSize == 0 {
		batchSize = testMax
	}

	// training set
	r := newReader(trainLabels, trainImages)
	s.Train = r.read(samples, batchSize, trainMax)
	s.NumInputs = int(r.image.Width * r.image.Height)
	s.NumOutputs = numOutputs
	s.MaxSamples = s.Train.NumSamples

	// validation set
	r.seek(trainMax)
	s.Valid = r.read(samples, batchSize, testMax)
	r.close()
	if r.err != nil {
		return s, r.err
	}

	// test set
	r = newReader(testLabels, testImages)
	s.Test = r.read(samples, batchSize, testMax)
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
	if Debug {
		fmt.Printf("open %s : %+v\n", labelFile, r.label)
	}
	if r.fimg, r.err = os.Open(base + imageFile); r.err != nil {
		return
	}
	if r.err = binary.Read(r.fimg, binary.BigEndian, &r.image); r.err != nil {
		return
	}
	if Debug {
		fmt.Printf("open %s : %+v\n", imageFile, r.image)
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
	if Debug {
		fmt.Printf("seek to image %d\n", n)
	}
}

// read an entire dataset
func (r *imageReader) read(samples, batchSize, maxImages int) *data.Data {
	d := new(data.Data)
	if samples == 0 || samples > maxImages {
		samples = maxImages
	}
	if batchSize > samples {
		batchSize = samples
	}
	nbatch := samples / batchSize
	d.Input = make([]blas.Matrix, nbatch)
	d.Output = make([]blas.Matrix, nbatch)
	d.Classes = make([]blas.Matrix, nbatch)
	for i := range d.Input {
		if r.err != nil {
			return nil
		}
		d.Input[i], d.Output[i], d.Classes[i] = r.readBatch(batchSize)
	}
	if Debug {
		fmt.Printf("read %d batches of %d images\n", nbatch, batchSize)
	}
	d.NumSamples = nbatch * batchSize
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
				idata[i*size*size+y*size+x] = rescale(image[(size-y-1)*size+x])
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

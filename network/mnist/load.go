package mnist

import (
	"encoding/binary"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
	"os"
)

// rescale intensity to value
func rescale(x uint8) float32 { return float32(x) / 255.0 }

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
	idata := make([]float32, n*size*size)
	odata := make([]float32, n*numOutputs)
	cdata := make([]float32, n)
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
			cdata[i] = float32(labels[i])
		}
	}
	in = blas.New(n, size*size).Load(blas.RowMajor, idata...)
	out = blas.New(n, numOutputs).Load(blas.RowMajor, odata...)
	class = blas.New(n, 1).Load(blas.RowMajor, cdata...)
	return
}

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
)

// register dataset when module is imported
func init() {
	data.Register["mnist"] = loader{}
}

type loader struct{}

// Load function loads and returns the iris dataset.
// samples is maxiumum number of records to load from each dataset if non-zero.
// TODO implement batch loading
func (loader) Load(samples, batchSize int) (s *data.Dataset, err error) {
	s = new(data.Dataset)
	s.OutputToClass = classify{}
	// test set
	images, size := readData(testLabels, testImages, 0, -1, samples)
	s.Test, s.NumInputs, s.NumOutputs = parseData(images, size)
	s.MaxSamples = s.Test.NumSamples
	// training set
	images, _ = readData(trainLabels, trainImages, 0, 50000, samples)
	s.Train, _, _ = parseData(images, size)
	s.MaxSamples = max(s.MaxSamples, s.Train.NumSamples)
	// validation set
	images, _ = readData(trainLabels, trainImages, 50000, -1, samples)
	s.Valid, _, _ = parseData(images, size)
	s.MaxSamples = max(s.MaxSamples, s.Valid.NumSamples)
	return
}

type classify struct{}

func (classify) Apply(out, class blas.Matrix) blas.Matrix {
	return class.MaxCol(out)
}

type image struct {
	data  []byte
	label byte
}

// load data from file
func readData(labelFile, imageFile string, from, to, samples int) ([]image, int) {
	var f *os.File
	var err error
	// labels
	var labelDef struct{ Magic, Num int32 }
	if f, err = os.Open(base + labelFile); err != nil {
		panic(fmt.Sprintf("error reading labels: %s", err))
	}
	binary.Read(f, binary.BigEndian, &labelDef)
	maxSamples := int(labelDef.Num) - from
	if to > 0 && to-from < maxSamples {
		maxSamples = to - from
	}
	if samples == 0 || samples > maxSamples {
		samples = maxSamples
	}
	fmt.Printf("read %d images from %s starting from %d\n", samples, imageFile, from)
	labels := make([]byte, samples)
	if from > 0 {
		fmt.Println("seek to image", from)
		if _, err = f.Seek(int64(from), 1); err != nil {
			panic(fmt.Sprintf("error in seek on %s: %s", labelFile, err))
		}
	}
	f.Read(labels)
	f.Close()
	// images
	var imageDef struct{ Magic, Num, Width, Height int32 }
	if f, err = os.Open(base + imageFile); err != nil {
		panic(fmt.Sprintf("error reading images: %s", err))
	}
	binary.Read(f, binary.BigEndian, &imageDef)
	images := make([]image, samples)
	size := imageDef.Width * imageDef.Height
	if from > 0 {
		if _, err = f.Seek(int64(from)*int64(size), 1); err != nil {
			panic(fmt.Sprintf("error in seek on %s: %s", labelFile, err))
		}
	}
	for i := 0; i < samples; i++ {
		images[i].label = labels[i]
		images[i].data = make([]byte, size)
		f.Read(images[i].data)
	}
	f.Close()
	return images, int(imageDef.Width)
}

// convert data set to neural net inputs and outputs
func parseData(img []image, size int) (d *data.Data, nin, nout int) {
	d = new(data.Data)
	d.NumSamples = len(img)
	nin = len(img[0].data)
	nout = numOutputs
	in := make([]float64, nin*d.NumSamples)
	out := make([]float64, nout*d.NumSamples)
	class := make([]float64, d.NumSamples)
	for i, image := range img {
		for iy := 0; iy < size; iy++ {
			for ix := 0; ix < size; ix++ {
				in[i*nin+iy*size+size-ix-1] = float64(image.data[ix*size+iy]) / 255
			}
		}
		for j := 0; j < nout; j++ {
			if int(image.label) == j {
				out[i*nout+j] = 1
			}
			class[i] = float64(image.label)
		}
	}
	d.Input = []blas.Matrix{blas.New(d.NumSamples, nin).Load(blas.RowMajor, in...)}
	d.Output = []blas.Matrix{blas.New(d.NumSamples, nout).Load(blas.RowMajor, out...)}
	d.Classes = []blas.Matrix{blas.New(d.NumSamples, 1).Load(blas.RowMajor, class...)}
	return
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

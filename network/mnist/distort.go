package mnist

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
	"math"
)

const (
	size         = 28
	size2        = size * size
	severity     = 1.0
	scale        = 0.15
	rotate       = 15.0 * math.Pi / 180.0
	elasticSigma = 8.0
	elasticScale = 0.25
	kernelSize   = 21
)

// supported types of distortion
const (
	Scale   = 1
	Rotate  = 2
	Elastic = 4
)

type Loader struct {
	filter blas.Filter
	img    blas.Image
	uimg   blas.Image
	xscale blas.Matrix
	yscale blas.Matrix
	angle  blas.Matrix
	kernel blas.Matrix
	unifx  blas.Matrix
	unify  blas.Matrix
	dx     blas.Matrix
	dy     blas.Matrix
	debug  bool
}

func (l *Loader) Debug(on bool) {
	l.debug = on
}

func (l *Loader) init(batch int) {
	l.img = blas.NewImage(size, size, batch, 1)
	// arrays for scaling and rotation
	l.xscale = blas.New(1, batch)
	l.yscale = blas.New(1, batch)
	l.angle = blas.New(1, batch)
	// convolution kernel and elastic distortion arrays
	l.filter = blas.NewFilter(kernelSize)
	l.kernel = blas.GaussianKernel(kernelSize, kernelSize, elasticSigma)
	//l.kernel.SetFormat("%4.2f")
	//fmt.Printf("convolve kernel:\n%s\n", l.kernel)
	l.unifx = blas.New(batch, size2)
	l.unify = blas.New(batch, size2)
	l.uimg = blas.NewImage(size, size, batch, 2)
	// distortion map in x and y dimensions
	l.dx = blas.New(batch, size2)
	l.dy = blas.New(batch, size2)
}

// DistortTypes returns the supported types of distortions
func (l *Loader) DistortTypes() []network.Distortion {
	return []network.Distortion{
		{Scale, "scale"},
		{Rotate, "rotate"},
		{Elastic, "elastic"},
	}
}

// Distort method is used to apply distortions to a batch of images.
// mask of -1 indicates all distortions are to be applied.
func (l *Loader) Distort(in, out blas.Matrix, mask int, severity float32) {
	//fmt.Println("distort: severity=", severity, "mask=", mask, "batch=", in.Rows())
	batch := in.Rows()
	if l.img == nil {
		l.init(batch)
	}
	l.img.Import(in)
	if (mask < 0 || mask&Elastic != 0) && elasticScale != 0 {
		l.unifx.Random(-1, 1)
		l.unify.Random(-1, 1)
		l.uimg.Import(l.unifx, l.unify)
		l.filter.Apply(l.uimg, l.kernel, l.dx, l.dy)
		l.dx.Scale(elasticScale * severity)
		l.dy.Scale(elasticScale * severity)
		if l.debug {
			l.dx.SetFormat("%5.2f")
			l.dy.SetFormat("%5.2f")
			cen := size2/2 + size/2
			fmt.Printf("elastic %s %s\n", l.dx.Row(0, 1).Col(cen-2, cen+3), l.dy.Row(0, 1).Col(cen-2, cen+3))
		}
	} else {
		l.dx.Set(0)
		l.dy.Set(0)
	}
	if (mask < 0 || mask&Rotate != 0) && rotate != 0 {
		l.angle.Random(-severity*rotate, severity*rotate)
		if l.debug {
			fmt.Printf("rotate  %s\n", l.angle.Col(0, 1))
		}
		l.img.Rotate(l.angle, l.dx, l.dy)
	}
	if (mask < 0 || mask&Scale != 0) && scale != 0 {
		l.xscale.Random(-severity*scale, severity*scale)
		l.yscale.Random(-severity*scale, severity*scale)
		if l.debug {
			fmt.Printf("scale   %s %s\n", l.xscale.Col(0, 1), l.yscale.Col(0, 1))
		}
		l.img.Scale(l.xscale, l.yscale, l.dx, l.dy)
	}
	l.img.Export(l.dx, l.dy, out)
}

func (l *Loader) Release() {
	if l.img != nil {
		l.img.Release()
		l.xscale.Release()
		l.yscale.Release()
		l.angle.Release()
		l.kernel.Release()
		l.unifx.Release()
		l.unify.Release()
		l.uimg.Release()
		l.dx.Release()
		l.dy.Release()
	}
	l.img = nil
}

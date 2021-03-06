package main

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/mplot"
	"os"
	"runtime"
)

const (
	width   = 800
	height  = 800
	rows    = 6
	cols    = 6
	imgSize = 28
)

func main() {
	runtime.LockOSThread()
	blas.Init(blas.Native32)
	d, err := data.Load("mnist", 10000)
	checkErr(err)
	fmt.Printf("loaded digits: %d training, %d test, %d validation\n",
		d.Train.NumSamples, d.Test.NumSamples, d.Valid.NumSamples)
	dataset := d.Test
	dataset.Input.SetFormat("%.0f")

	window, err := mplot.NewWindow(width, height, "digits")
	checkErr(err)
	plt := make([]*mplot.Plot, rows*cols)
	mat := make([]blas.Matrix, rows*cols)
	labels := dataset.Classes.Data(blas.RowMajor)
	for i := range plt {
		mat[i] = blas.New(imgSize, imgSize)
		plt[i] = mplot.New()
		plt[i].X.Tick.Length = 0
		plt[i].Y.Tick.Length = 0
		plt[i].X.Tick.Label.Font.Size = 0
		plt[i].Y.Tick.Label.Font.Size = 0
		plt[i].Add(mplot.NewImage(mat[i]))
	}
	page := 0

	go func() {
		for {
			for i := range plt {
				ix := page*rows*cols + i
				mat[i].Copy(dataset.Input.Row(ix, ix+1).Reshape(imgSize, imgSize, true))
				//fmt.Printf("%d: \n%s\n", i, mat[i])
				plt[i].Title.Text = fmt.Sprintf("test %d: %.0f", ix, labels[ix])
			}
			fmt.Print("hit return for next page")
			fmt.Fscanln(os.Stdin)
			page++
		}
	}()

	for !window.ShouldClose() {
		window.Draw(rows, cols, plt...)
	}
}

func checkErr(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

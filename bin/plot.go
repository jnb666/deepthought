package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"time"

	"github.com/jnb666/deepthought/m32"
	"github.com/jnb666/deepthought/mplot"
)

const (
	width  = 640
	height = 640
	points = 100
	maxy   = 140
)

func randval(i int) float32 {
	return float32(i) + 30*rand.Float32()
}

func checkErr(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func main() {
	runtime.LockOSThread()
	window, err := mplot.NewWindow(width, height, "viewer")
	checkErr(err)

	p, err := mplot.New()
	checkErr(err)
	p.Title.Text = "Plot example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	v1 := m32.NewVector(0, points)
	v2 := m32.NewVector(0, points)
	v3 := m32.NewVector(0, points)

	mplot.AddLines(p,
		mplot.NewLine(v1, "first", 0, maxy),
		mplot.NewLine(v2, "second", 0, maxy),
		mplot.NewLine(v3, "third", 0, maxy),
	)

	go func() {
		for i := 0; i < points; i++ {
			v1.Append(randval(i))
			v2.Append(randval(i))
			v3.Append(randval(i))
			time.Sleep(100 * time.Millisecond)
		}
	}()
	for !window.ShouldClose() {
		window.Update(p)
	}
}

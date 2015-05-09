package main

import (
	"flag"
	"fmt"
	"os"
	"text/template"
)

type Template struct {
	Struct string
	Type   string
}

func checkErr(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func main() {
	var t Template
	flag.StringVar(&t.Struct, "struct", "", "struct name")
	flag.StringVar(&t.Type, "type", "", "type name")
	flag.Parse()
	tmpl, err := template.ParseFiles(flag.Arg(0))
	checkErr(err)
	f, err := os.Create(flag.Arg(1))
	checkErr(err)
	err = tmpl.Execute(f, t)
	checkErr(err)
}

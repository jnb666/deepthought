// Package config converts a config struct to and from string format and persists to file.
package config

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strconv"
)

var ConfigDir = os.Getenv("HOME") + "/.deepthought"

// check ConfigDir exists - if not then create it
func checkConfigDir() error {
	info, err := os.Stat(ConfigDir)
	if err == nil && !info.IsDir() {
		err = fmt.Errorf("%s: not a directory", ConfigDir)
	} else if os.IsNotExist(err) {
		err = os.Mkdir(ConfigDir, 0755)
	}
	return err
}

// construct config file name
func fileName(name string) string {
	return ConfigDir + "/" + name + ".cfg"
}

// Print the config to stdout
func Print(cfg interface{}) {
	for _, key := range Keys(cfg) {
		fmt.Printf("%12s : %v\n", key, Get(cfg, key))
	}
}

// Update values in config struct cfg to in
func Update(cfg, in interface{}) {
	for _, key := range Keys(cfg) {
		Set(cfg, key, Get(in, key))
	}
}

// Save config struct to disk
func Save(cfg interface{}, name string) error {
	if err := checkConfigDir(); err != nil {
		return err
	}
	buf, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	file := fileName(name)
	fmt.Println("save config to", file)
	return ioutil.WriteFile(file, buf, 0644)
}

// Load struct from disk
func Load(cfg interface{}, name string) error {
	if err := checkConfigDir(); err != nil {
		return err
	}
	file := fileName(name)
	fmt.Println("load config from", file)
	r, err := os.Open(file)
	if err != nil {
		return err
	}
	buf, err := ioutil.ReadAll(r)
	if err != nil {
		return err
	}
	return json.Unmarshal(buf, cfg)
}

// Return an array of strings with the struct keys
func Keys(cfg interface{}) []string {
	s := reflect.ValueOf(cfg).Elem()
	keys := make([]string, s.NumField())
	for i := range keys {
		keys[i] = s.Type().Field(i).Name
	}
	return keys
}

// Get config value as a string
func Get(cfg interface{}, key string) string {
	s := reflect.ValueOf(cfg).Elem()
	fld := s.FieldByName(key)
	if fld.IsValid() {
		return fmt.Sprint(fld.Interface())
	}
	return ""
}

// Set config value from a string
func Set(cfg interface{}, key, value string) error {
	s := reflect.ValueOf(cfg).Elem()
	fld := s.FieldByName(key)
	if fld.IsValid() {
		t := fld.Type().Kind()
		switch t {
		case reflect.String:
			fld.SetString(value)
		case reflect.Int, reflect.Int64:
			val, err := strconv.ParseInt(value, 10, 0)
			if err != nil {
				return fmt.Errorf("Config: invalid value for %s: %s", key, err)
			}
			fld.SetInt(val)
		case reflect.Float32:
			val, err := strconv.ParseFloat(value, 32)
			if err != nil {
				return fmt.Errorf("Config: invalid value for %s: %s", key, err)
			}
			fld.SetFloat(val)
		case reflect.Float64:
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return fmt.Errorf("Config: invalid value for %s: %s", key, err)
			}
			fld.SetFloat(val)
		case reflect.Bool:
			val, err := strconv.ParseBool(value)
			if err != nil {
				return fmt.Errorf("Config: invalid value for %s: %s", key, err)
			}
			fld.SetBool(val)
		default:
			return fmt.Errorf("Config: unsupported type %v", t)
		}
	}
	return nil
}

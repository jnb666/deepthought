// Package config converts a config struct to and from string format and persists to file.
package config

import (
	"fmt"
	"reflect"
	"strconv"
)

// Print the config to stdout
func Print(cfg interface{}) {
	s := reflect.ValueOf(cfg).Elem()
	for i := 0; i < s.NumField(); i++ {
		fmt.Printf("%12s : %v\n", s.Type().Field(i).Name, s.Field(i).Interface())
	}
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
		switch fld.Type().Kind() {
		case reflect.String:
			fld.SetString(value)
		case reflect.Int, reflect.Int64:
			val, err := strconv.ParseInt(value, 10, 0)
			if err != nil {
				return fmt.Errorf("FromMap: invalid value for %s: %s", key, err)
			}
			fld.SetInt(val)
		case reflect.Float64:
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return fmt.Errorf("FromMap: invalid value for %s: %s", key, err)
			}
			fld.SetFloat(val)
		case reflect.Bool:
			val, err := strconv.ParseBool(value)
			if err != nil {
				return fmt.Errorf("FromMap: invalid value for %s: %s", key, err)
			}
			fld.SetBool(val)
		}
	}
	return nil
}

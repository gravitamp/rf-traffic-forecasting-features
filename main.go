// Here is the process for Random Forest: 1. You still have to transform your data
// 2. You still have to test for stationarity
// 3. You have to think about creating a bunch of useful features like season, time of day, t-1, t-7, t-14,
// split weeks, holidays, features that go into all machine learning models
// 4. Set up cross validation (train, test)
// 5. Optimize with gridsearch or kfold
// 6. Pick parameters, then run a model
// 7. Look at results

package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

var (
	vehiclestrain []float64
	datetrain     []float64
	vehiclestest  []float64
	datetest      []float64
	weathertrain  []float64
	temptrain     []float64
	holidaytrain  []float64
	cloudstrain   []float64
	weathertest   []float64
	temptest      []float64
	holidaytest   []float64
	cloudstest    []float64
	forecast      []float64
)

const (
	CLEAR        = 1.0
	CLOUDS       = 2.0
	HAZE         = 3.0
	MIST         = 4.0
	FOG          = 5.0
	DRIZZLE      = 6.0
	RAIN         = 7.0
	SNOW         = 8.0
	THUNDERSTORM = 9.0
)

func main() {
	// setup and split dataset
	setupData("Metro_Interstate_Traffic_Volume.csv")

	// // split into input and output columns
	count := len(vehiclestrain) - 3
	train_inputs := make([][]interface{}, count)
	train_targets := make([]float64, count)

	for i := 0; i < count; i++ {
		train_inputs[i] = []interface{}{vehiclestrain[i], vehiclestrain[i+1], vehiclestrain[i+2],
			weathertrain[i+3], temptrain[i+3],
			cloudstrain[i+3], holidaytrain[i+3], datetrain[i+3]}
		train_targets[i] = vehiclestrain[i+3]
	}

	//fit model
	forest := BuildForest(train_inputs, train_targets, count, len(train_inputs), 8)
	// fmt.Println(forest)

	//testing
	y := []interface{}{vehiclestest[0], vehiclestest[1], vehiclestest[2],
		weathertest[3], temptest[3], cloudstest[3], holidaytest[3], datetest[3]}

	fmt.Println(y, "predicted: ", forest.Predicate(y), "test: ", vehiclestest[3])

	count2 := len(vehiclestest) - 3
	for i := 0; i < count2; i++ {
		predict := []interface{}{vehiclestest[i], vehiclestest[i+1], vehiclestest[i+2],
			weathertest[i+3], temptest[i+3], cloudstest[i+3], holidaytest[i+3], datetest[i+3]}
		forecast = append(forecast, forest.Predicate(predict))

	}

	//add plot
	p := plot.New()
	p.Title.Text = "Traffic Volume Forecasting"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	err := plotutil.AddLinePoints(p,
		"Train", makePoints(vehiclestrain, datetrain),
		"Test", makePoints(vehiclestest, datetest),
		"Predict", makePoints(forecast, datetest[3:]))
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(10*vg.Inch, 4*vg.Inch, "plot.png"); err != nil {
		panic(err)
	}

}

func setupData(file string) {
	f, err := os.Open(file)
	if err != nil {
		return
	}
	csvReader := csv.NewReader(f)
	csvData, err := csvReader.ReadAll()

	//read without header
	for i := 1; i < len(csvData); i++ {
		cuaca := CLEAR
		switch csvData[i][5] {
		case "Clear":
			cuaca = CLEAR
			break
		case "Clouds":
			cuaca = CLOUDS
			break
		case "Haze":
			cuaca = HAZE
			break
		case "Mist":
			cuaca = MIST
			break
		case "Fog":
			cuaca = FOG
			break
		case "Drizzle":
			cuaca = DRIZZLE
			break
		case "Rain":
			cuaca = RAIN
			break
		case "Snow":
			cuaca = SNOW
			break
		case "Thunderstorm":
			cuaca = THUNDERSTORM
			break
		}
		days := 0.0
		if csvData[i][0] != "None" {
			days = 1.0
		}
		val, _ := strconv.ParseFloat(csvData[i][8], 64)
		temp, _ := strconv.ParseFloat(csvData[i][1], 64)
		cloud, _ := strconv.ParseFloat(csvData[i][4], 64)

		layouts := "2006-01-02 15:04:05"
		t, err := time.Parse(layouts, csvData[i][7])
		if err != nil {
			fmt.Println(err)
		}

		//don't split randomly
		if float64(i) < (float64(len(csvData)) * 0.9) {
			vehiclestrain = append(vehiclestrain, val)
			weathertrain = append(weathertrain, cuaca)
			temptrain = append(temptrain, temp)
			holidaytrain = append(holidaytrain, days)
			cloudstrain = append(cloudstrain, cloud)
			datetrain = append(datetrain, float64(t.Unix()))
		} else {
			vehiclestest = append(vehiclestest, val)
			weathertest = append(weathertest, cuaca)
			temptest = append(temptest, temp)
			holidaytest = append(holidaytest, days)
			cloudstest = append(cloudstest, cloud)
			datetest = append(datetest, float64(t.Unix()))
		}

	}
}

func makePoints(data []float64, date []float64) plotter.XYs {
	pts := make(plotter.XYs, len(data))
	for i := range pts {
		pts[i].X = date[i]
		pts[i].Y = data[i]
	}
	return pts
}

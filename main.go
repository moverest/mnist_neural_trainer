// Copyright 2016 Clément Martinez

package main

import (
	"flag"
	"log"

	"github.com/moverest/mnist"
	"github.com/moverest/neuralmnist"
	"github.com/moverest/neuralnet"
)

const (
	dirFlagHelp = "folder containing the MNIST data base files"
	defaultDir  = "."
	dirFlagName = "dir"

	learningRateHelp      = "neural network learning rate"
	defaultLearningRate   = 3.
	learningRateFlageName = "learningRate"

	epochSizeHelp     = "epoch size"
	defaultEpochSize  = 10
	epochSizeFlagName = "epochSize"

	numEpochHelp     = "number of epoch to be performed"
	defaultNumEpoch  = 10
	numEpochFlagName = "numEpoch"

	middleLayerSizeHelp     = "middle layer size"
	defaultMiddleLayerSize  = 30
	middleLayerSizeFlagName = "midSize"

	loadNetFileNameHelp    = "load network file"
	defaultLoadNetFileName = ""
	loadNetFileNameFlag    = "load"

	saveNetFileNameHelp   = "file in which to save the trained network"
	defautSaveNetFileName = ""
	saveNetFileNameFlag   = "out"
)

func main() {
	var databaseDir, loadNetFileName, saveNetFileName string
	var learningRate float64
	var epochSize, numEpoch, middleLayerSize int

	flag.StringVar(&databaseDir, dirFlagName, defaultDir, dirFlagHelp)
	flag.StringVar(&loadNetFileName, loadNetFileNameFlag, defaultLoadNetFileName, loadNetFileNameHelp)
	flag.StringVar(&saveNetFileName, saveNetFileNameFlag, defautSaveNetFileName, saveNetFileNameHelp)
	flag.Float64Var(&learningRate, learningRateFlageName, defaultLearningRate, learningRateHelp)
	flag.IntVar(&epochSize, epochSizeFlagName, defaultEpochSize, epochSizeHelp)
	flag.IntVar(&numEpoch, numEpochFlagName, defaultNumEpoch, numEpochHelp)
	flag.IntVar(&middleLayerSize, middleLayerSizeFlagName, defaultMiddleLayerSize, middleLayerSizeHelp)
	flag.Parse()

	mnistTraining, mnistTest, err := mnist.Load(databaseDir)
	if err != nil {
		log.Fatal("main: ", err)
	}

	training := neuralnetminst.ConvertSet(mnistTraining)
	test := neuralnetminst.ConvertSet(mnistTest)

	log.Printf("MNIST database loaded (training:%v, test:%v)", training.Count(),
		test.Count())

	log.Println("Learning rate:", learningRate)
	log.Println("Epoch size:", epochSize)
	log.Println("Number of epoch:", numEpoch)

	var net *neuralnet.Network

	if loadNetFileName == defaultLoadNetFileName {
		net = neuralnet.New([]int{784, middleLayerSize, 10})
		net.SeedRand()
		net.Randomize()
		log.Println("Random network created:", net.Sizes)
	} else {
		net, err = neuralnet.LoadFile(loadNetFileName)
		if err != nil {
			log.Fatalln("mnist_neural_trainer:", err)
		}
		log.Println("Network loaded:", loadNetFileName, net.Sizes)

		n := len(net.Sizes)
		if n == 0 || net.Sizes[0] != 28*28 || net.Sizes[n-1] != 10 {
			log.Fatalln("mnist_neural_trainer: loaded neural network is not conform")
		}
	}

	net.SGD(training, test, Evaluate, numEpoch, epochSize, learningRate)

	if saveNetFileName != "" {
		err := net.SaveFile(saveNetFileName)
		if err != nil {
			log.Fatalln("mnist_neural_trainer: error while saving the network to a file:",
				err)
		}
		log.Println("Neural network saved:", saveNetFileName)
	}
}

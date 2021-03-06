import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.1
import GoExtensions 1.0

ApplicationWindow {
	id: root; title: "deepthought"; color: "lightgray"; 
	x: 50; y: 50
	minimumWidth: 800; minimumHeight: 900
	onClosing: ctrl.send("quit", "")

	TabView{
		id: tab		
		objectName: "tabs"
		anchors.left: parent.left; anchors.right: parent.right
		height: 450
		Component.onCompleted: {
			addTab("train", tab1)
			addTab("test", tab2)
			addTab("options", tab3)
		}

		Component {
			id: tab1
			ColumnLayout {	
				RowLayout {
					spacing: 20
					Button { text: "start"; onClicked: ctrl.send("start", "") }
					Button { text: "step"; onClicked: ctrl.send("step", "") }
					Button {
						objectName: "runButton"; text: "run"; checkable: true
						onClicked: ctrl.send("run", checked ? "start" : "stop");
					}
					Button { text: "stop"; onClicked: ctrl.send("stop", "") }
					Label { text: "data set:" }
					ComboBox {
						objectName: "modelList"
						width: 120
						model: ListModel {
							id: dataSets
							{{ range .Datasets }}
							ListElement { text: "{{ . }}" }
							{{ end }}
						}
						currentIndex: {{ .Index }}
						onActivated: ctrl.select(dataSets.get(index).text)
					}
					Label { objectName: "runLabel" }
				}
				Plots{
					id: plot; objectName: "plotControl"
					width: 800; height: 800
					background: "black"; color: "white"
					grid: true; gridColor: "#404040"
					Timer {
						interval: 100; running: true; repeat: true
						onTriggered: plot.update()
					}
				}
				RowLayout {
					ExclusiveGroup { id: plotGroup }
					{{ range .Plots }}
					RadioButton { 
						text: "{{ .Name }}"; exclusiveGroup: plotGroup; checked: {{ .Selected }}
						onClicked: plot.select({{ .Index }})
					}
					{{ end }}
					Button { text: "print stats"; onClicked: ctrl.send("stats", "print") }
					Button { text: "clear stats"; onClicked: ctrl.send("stats", "clear") }
				}
			}
		}

		Component {
			id: tab2
			ColumnLayout {	
				RowLayout {
					spacing: 20
					Button { text: "first"; onClicked: net.first() }					
					Button { text: "<< prev"; onClicked: net.prev() }
					Label { objectName: "testLabel"}
					Button { text: "next >>"; onClicked: net.next() }
					Button { 
						id: distort; text: "distort"; checkable: true; 
						onClicked: net.distort(checked, distortMode.currentIndex) 
					}
					ComboBox {
						id: distortMode
						model: ListModel {
							id: distortList
							objectName: "distortList"
							function reset() {
								distortMode.currentIndex = 0
								distortList.clear()
							}
							function addItem(t) {
								distortList.append({ text: t })
							}
						}
						onActivated: net.distort(distort.checked, index)
					}					
				}
				RowLayout {
					Network {
						id: net; objectName: "netControl"
						width: 800; height: 800
						background: "#202020"; color: "white"
					}
				}
				RowLayout {
					spacing: 20					
					CheckBox { 
						text: "compact view"; onClicked: net.compact(checked) 
					}
					CheckBox {
						id: showErrors 
						text: "only show errors"; onClicked: net.filter(checked, filter.currentIndex-1) 
					}
					Label { text: "filter:" }
					ComboBox { 
						id: filter
						model: ListModel {
							id: filterList
							objectName: "filterList"
							function reset() {
								filter.currentIndex = 0
								filterList.clear()
							}
							function addItem(t) {
								filterList.append({ text: t })
							}
						}
						onActivated: net.filter(showErrors.checked, index-1)
					}
				}
			}
		}

		Component {
			id: tab3
			GridLayout {
				columns: 3
				columnSpacing: 5; rowSpacing: 5
				Label { 
					text: "number of runs"
					anchors.right: runs.left; anchors.rightMargin: 10
				}
				TextField {
					id: runs; objectName: "MaxRuns"
					validator: IntValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Button { 
					text: "save"; onClicked: cfg.save(cfg.model) 
				}
				Label { 
					text: "max epoch"
					anchors.right: epochs.left; anchors.rightMargin: 10
				}
				TextField {
					id: epochs; objectName: "MaxEpoch"
					validator: IntValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Button { 
					text: "load"; onClicked: cfg.load(cfg.model)  
				}
				Label { 
					text: "learning rate"
					anchors.right: eta.left; anchors.rightMargin: 10
				}
				TextField {
					id: eta; objectName: "LearnRate"
					validator: DoubleValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Button { 
					text: "default"; onClicked: cfg.default(cfg.model)  
				}				
				Label {
					text: "weight decay"
					anchors.right: lambda.left;	anchors.rightMargin: 10
				}
				TextField { 
					id: lambda;	objectName: "WeightDecay"
					validator: DoubleValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Button { 
					text: "print"; onClicked: cfg.print()  
				}					
				Label {
					text: "momentum"
					anchors.right: mom.left; anchors.rightMargin: 10
				}
				TextField { 
					id: mom; objectName: "Momentum"
					validator: DoubleValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Label { 
					Layout.rowSpan: 8
				}
				Label {
					text: "threshold"
					anchors.right: threshold.left; anchors.rightMargin: 10
				}
				TextField { 
					id: threshold; objectName: "Threshold"
					validator: DoubleValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Label {
					text: "batch size"
					anchors.right: batch.left; anchors.rightMargin: 10
				}
				TextField { 
					id: batch; objectName: "BatchSize"
					validator: IntValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Label {
					text: "stop after"
					anchors.right: stop.left; anchors.rightMargin: 10
				}
				TextField { 
					id: stop; objectName: "StopAfter"
					validator: IntValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Label {
					text: "log every"
					anchors.right: log.left; anchors.rightMargin: 10
				}
				TextField { 
					id: log; objectName: "LogEvery"
					validator: IntValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
				Label {
					text: "sampler"
					anchors.right: sampler.left; anchors.rightMargin: 10
				}
				ComboBox { 
					id: sampler; objectName: "Sampler"
					model: ["uniform", "random"]
					onActivated: cfg.set(objectName, model[index])
				}
				Label {
					text: "distortion"
					anchors.right: distortion.left; anchors.rightMargin: 10
				}									
				TextField { 
					id: distortion; objectName: "Distortion"
					validator: DoubleValidator{}
					onTextChanged: cfg.set(objectName, text)
				}
			}
		}
	}
}